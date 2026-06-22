# NOAA CUDEM Sources

Use this reference when updating `cudem-bathy` source logic.

## CUDEM Scope

NOAA CUDEM is a U.S. coastal and territory product, not a global bathymetry source. CUDEM-only commands intentionally fail with a no-coverage report when a bbox is outside the indexed CUDEM tiles. Fallback-enabled commands can fill gaps with NOAA NBS BlueTopo candidates, NOAA Coastal Relief Models, and then ETOPO 2022.

NOAA describes CUDEM as 0.25-degree tiled coastal DEMs. The 1/9 arc-second product integrates topography and bathymetry; the coarser products are bathymetry/topobathymetry tiers used farther from the coast or in older regional products. CUDEM is distributed in NetCDF and GeoTIFF formats, with coordinates in decimal degrees, horizontal datum NAD83, vertical datum NAVD88 where provided, and vertical units in meters.

## Machine Sources

Primary THREDDS root:

```text
https://www.ngdc.noaa.gov/thredds/catalog/tiles/catalog.xml
```

Useful root-level THREDDS collections:

```text
https://www.ngdc.noaa.gov/thredds/catalog/tiles/tiled_19as/catalog.xml
https://www.ngdc.noaa.gov/thredds/catalog/tiles/tiled_13as/catalog.xml
https://www.ngdc.noaa.gov/thredds/catalog/tiles/tiled_1as/catalog.xml
https://www.ngdc.noaa.gov/thredds/catalog/tiles/tiled_3as/catalog.xml
```

The nested `tiles/cudem/tiled_*` catalogs may respond, but they were effectively empty during planning. Do not prefer them for v1.

Digital Coast ninth-arc-second bulk root:

```text
https://chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/
```

Digital Coast CUDEM URL lists:

```text
https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_ninth_Topobathy_2014_8483/urllist8483.txt
https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/dem/NCEI_third_Topobathy_2014_8580/urllist8580.txt
```

Use the URL lists when broad regional coverage is needed. The SE-AK case uses
1/9 arc-second CUDEM where available near Icy Strait and 1/3 arc-second CUDEM
for broader southern SE-AK and Sumner Strait coverage.

Known important regional folders:

```text
AK
wash_pugetsound
```

These folders expose HTTPS GeoTIFF tiles hosted under `noaa-nos-coastal-lidar-pds.s3.amazonaws.com`.

## NBS And Fallback Sources

Use this priority when the user requests fallback-enabled bathymetry:

```text
CUDEM + NOAA NBS BlueTopo candidates -> NOAA Coastal Relief Model -> ETOPO 2022
```

Conservative source-priority mode keeps CUDEM before BlueTopo:

```text
CUDEM -> NBS BlueTopo -> NOAA Coastal Relief Model -> ETOPO 2022
```

Finest mode lets every selected local source compete by native resolution:

```text
finest(CUDEM, NBS BlueTopo, NOAA Coastal Relief Model, ETOPO 2022)
```

This means ETOPO 2022 15 arc-second can outrank the legacy Southern Alaska CRM
24 arc-second in gaps where no finer CUDEM/NBS source is available. Use
`source-priority` when the scientific preference is source family before native
grid spacing.

NOAA NBS open-data registry:

```text
https://registry.opendata.aws/noaa-bathymetry/
```

NOAA NBS AWS bucket:

```text
s3://noaa-ocs-nationalbathymetry-pds
https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/
```

BlueTopo tile-scheme GeoPackages:

```text
https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/?list-type=2&prefix=BlueTopo/_BlueTopo_Tile_Scheme/
```

Important BlueTopo details:

- The tile scheme advertises `GeoTIFF_Link`, `RAT_Link`, `Delivered_Date`, `Resolution`, `UTM`, checksums, and tile polygons.
- BlueTopo GeoTIFFs are commonly projected UTM compound CRS rasters; do not sample them as lon/lat rasters.
- BlueTopo bands commonly include `Elevation`, `Uncertainty`, and `Contributor`.
- Use `/vsicurl` or HTTPS reads for selected tiles; do not download all BlueTopo tiles over a broad bbox.
- The current implementation treats BlueTopo as a first-class `nbs_bluetopo` source and records uncertainty/contributor for mesh-node sampling where available.
- Mixed CUDEM/BlueTopo vertical datums are not harmonized by this skill.

NOAA NBS S-102 is documented as a future source because it is HDF5/S-102 rather
than directly sampled GeoTIFF:

```text
https://noaa-s102-pds.s3.amazonaws.com/README.html
s3://noaa-s102-pds
```

For SE Alaska, the S-102 bucket has an Alaska/Southeast folder, but it is not yet
an automatic fetch source in this skill.

NOAA CRM product page:

```text
https://www.ncei.noaa.gov/products/coastal-relief-model
```

Current 1 arc-second CRM THREDDS catalog:

```text
https://www.ngdc.noaa.gov/thredds/catalog/crm/cudem/catalog.xml
```

Legacy CRM catalog, including Southern Alaska:

```text
https://www.ngdc.noaa.gov/thredds/catalog/crm/catalog.xml
```

Important CRM details:

- Current CRM volumes are advertised as 1 arc-second where available.
- Southern Alaska CRM is available as `crm_southak.nc`, but it is 24 arc-second and uses 0-360 longitude coordinates.
- CRM vertical references vary; current products can use EGM2008, while older products may use Sea Level or Mean Sea Level.
- Do not treat CRM fallback as vertical-datum harmonization.

ETOPO 2022 product page:

```text
https://www.ncei.noaa.gov/products/etopo-global-relief-model
```

ETOPO 2022 15 arc-second bedrock-elevation NetCDF THREDDS catalog:

```text
https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/15s/15s_bed_elev_netcdf/catalog.xml
```

ETOPO 2022 15 arc-second surface-elevation NetCDF THREDDS catalog:

```text
https://www.ngdc.noaa.gov/thredds/catalog/global/ETOPO2022/15s/15s_surface_elev_netcdf/catalog.xml
```

Use ETOPO 2022 15 arc-second surface elevation as the global fallback for ordinary FVCOM bathymetry preprocessing. The `bed` catalog contains the bedrock-under-ice subset; it is not the complete global tile set. Outside ice sheets, ocean bathymetry is carried in the surface-elevation tiles.

## Tile Names

Example names:

```text
ncei13_n24x75_w080x50_2016v1.nc
ncei19_n47x50_w122x75_2023v1.tif
ncei19_n29X50_w085X00_2019v1.nc
```

Parsing rules:

- `ncei19` -> `tiled_19as` / 1/9 arc-second.
- `ncei13` -> `tiled_13as` / 1/3 arc-second.
- `ncei1` -> `tiled_1as`.
- `ncei3` -> `tiled_3as`.
- The latitude token is the north edge for northern-hemisphere tiles; `n24x75` spans `24.50` to `24.75`.
- The west longitude token is the west edge; `w080x50` spans `-80.50` to `-80.25`.
- Support both lowercase `x` and uppercase `X`.

## Required Smoke Regions

Run these as representative live tests:

```text
SE Alaska / Icy Strait-Juneau: -136.00 58.25 -135.50 58.49
Delaware Bay:                 -75.35 38.75 -74.95 39.10
Long Island Sound:            -73.30 40.80 -72.40 41.25
Puget Sound:                  -123.15 47.45 -122.35 48.05
```

## SE-AK Large Mesh Comparison

For `Resources/SE_AK_merged_complete_v5_latlon.2dm`, do not create one native
rectangular mosaic over the full mesh bbox. The mesh bbox is much larger than
the CUDEM-covered patches. Instead:

1. Build an expanded index using the ninth and third arc-second URL lists.
2. Select intersecting `tiled_19as` and `tiled_13as` GeoTIFFs.
3. Sample selected tiles directly to mesh nodes.
4. Assign 1/9 arc-second values first, then fill remaining covered nodes with
   1/3 arc-second values.
5. Compute positive-down depths and anomaly as
   `cudem_depth_m - original_depth_m`.

For fallback-enabled large-mesh comparisons:

1. Sample CUDEM first.
2. Sample NOAA NBS BlueTopo for nodes still missing bathymetry in `source-priority` mode.
3. In `finest` mode, order all selected source records by native resolution first, then source priority as a tie-breaker.
4. Preserve `best_source`, `best_source_resolution`, `best_source_resolution_m`, `best_uncertainty_m`, `best_contributor`, and `source_dataset` for every node.
5. Report mixed-datum warnings in the summary and plots.

## Regional Discovery

Use `research_bathy_sources.py` for an advisory, review-only regional source
scan. It records official/high-confidence alternatives, access method, rough
resolution, datum notes, endpoint status, and automation difficulty, but it does
not inject unvalidated products into the production stack.

Known SE-AK candidates to document:

```text
NOAA NBS BlueTopo
NOAA NBS S-102
NCEI BAG bathymetry ImageServer
NCEI multibeam mosaic ImageServer
NOAA Fisheries Alaska bathymetry service
SE Alaska 8 arc-second Coastal DEM
```
