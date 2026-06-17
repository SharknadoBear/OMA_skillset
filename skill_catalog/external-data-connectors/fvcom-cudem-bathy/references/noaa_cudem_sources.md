# NOAA CUDEM Sources

Use this reference when updating `fvcom-cudem-bathy` source logic.

## CUDEM Scope

NOAA CUDEM is a U.S. coastal and territory product, not a global bathymetry source. V1 of this skill intentionally fails with a no-coverage report when a bbox is outside the indexed CUDEM tiles.

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
