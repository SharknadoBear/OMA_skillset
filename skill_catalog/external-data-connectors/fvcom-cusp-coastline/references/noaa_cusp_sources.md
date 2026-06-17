# NOAA CUSP Sources

Use this reference when changing CUSP source handling.

## Source Choice

V1 uses official National Shoreline Data Explorer (NSDE) regional ZIP downloads.
The interactive NSDE map exposes these direct links, and the regional files are
small enough to cache locally and clip by bbox.

Primary NSDE page:

```text
https://nsde.ngs.noaa.gov/
```

NOAA CUSP metadata:

```text
https://www.fisheries.noaa.gov/inport/item/60812
```

## Regional ZIPs

```text
Alaska              https://geodesy.noaa.gov/dist_shoreline/Alaska.zip
Gulf of America     https://geodesy.noaa.gov/dist_shoreline/Gulf_Of_America.zip
North Atlantic      https://geodesy.noaa.gov/dist_shoreline/North_Atlantic.zip
Pacific Islands     https://geodesy.noaa.gov/dist_shoreline/Pacific_Islands.zip
Southeast Caribbean https://geodesy.noaa.gov/dist_shoreline/Southeast_Caribbean.zip
West                https://geodesy.noaa.gov/dist_shoreline/West.zip
Great Lakes         https://geodesy.noaa.gov/dist_shoreline/Great_Lakes.zip
Planned Projects    https://geodesy.noaa.gov/dist_shoreline/CUSP_IN_PROGRESS.zip
```

Do not use Planned Projects for production coastline fetches unless the user
explicitly asks for planned/in-progress CUSP work.

## ArcGIS FeatureServer

The NOAA Coastal Shoreline ArcGIS service is useful as a reference, but it is
not an implemented fallback in this skill:

```text
https://services.arcgis.com/rD2ylXRs80UroD90/arcgis/rest/services/NOAA_Coastal_Shoreline/FeatureServer
```

Do not prefer it in this skill. The official regional ZIPs are directly
advertised by NSDE and carry the full shapefile attributes needed for local
clipping.

## OSM Overpass Fallback

Use OSM only when the user explicitly requests fallback. Do not silently mix OSM
into a CUSP-only fetch.

The implemented fallback is a bbox Overpass query. By default the skill omits
the Overpass server-side timeout clause so long runs can wait:

```text
[out:json];
(
  way["natural"="coastline"](S,W,N,E);
);
out geom;
```

Only include `[timeout:N]` when the user explicitly passes
`--overpass-timeout-s N`. A value of `0` means omit the timeout.

Cache the raw Overpass JSON under the workspace. Convert ways to EPSG:4326 line
geometries and preserve `osm_id`, OSM tags, endpoint, query text, and
OpenStreetMap / ODbL attribution.

Do not download global OSM processed coastline, land, or water shapefiles for
v2 fallback. They are roughly 860--900 MB each and are too heavy for routine
bbox gap filling. The OSM processed coastline source remains a useful reference:

```text
https://osmdata.openstreetmap.de/data/coastlines.html
```

## Merge Rules

Keep all production CUSP segments. For OSM fallback, project both sources to a
local UTM CRS, buffer CUSP by 75 m, and retain only
`OSM geometry - CUSP buffer`. Explode retained OSM fragments, drop fragments
shorter than 100 m, and tag retained fallback as `osm_overpass`.

Do not remove an entire OSM way just because part of it overlaps CUSP. OSM
coastline ways can be many kilometers long, so whole-feature deletion can erase
valid gap-filling coastline.

GSHHG and Natural Earth are not active fallback policies in this skill. Treat
them only as diagnostic or future-work references if the user explicitly asks.

## CRS and Attributes

NOAA CUSP metadata identifies the product as NAD83 / EPSG:4269. For FVCOM
preprocessing outputs, convert geometries to EPSG:4326 while recording the
source CRS in metadata.

Preserve source fields when present:

```text
SOURCE_ID SRC_DATE HOR_ACC INFORM ATTRIBUTE VER_DATE SRC_RESOLU DATA_SOURC
EXT_METH DAT_SET_CR SRC_CITA FIPS_ALPHA NOAA_Regio GISDATE Shape_Leng
```

## Required Smoke Regions

```text
delaware_bay       -75.35 38.75 -74.95 39.10   North Atlantic
long_island_sound  -73.30 40.80 -72.40 41.25   North Atlantic
puget_sound       -123.15 47.45 -122.35 48.05  West
se_ak_icy_strait  -136.20 58.10 -134.80 58.60  Alaska
se_ak_sumner      -133.50 55.60 -131.70 56.40  Alaska
se_ak_icy_gap     -136.20 58.10 -135.612 58.60 Alaska + OSM fallback
se_ak_cross_sound -137.20 57.85 -136.15 58.55 Alaska + OSM fallback
```

## Data-Use Caveats

CUSP provides contemporary high-resolution shoreline for planning, mapping, and
modeling contexts. It is not for litigation, navigation, or authoritative
boundary determination. Source vintage and accuracy vary by segment.
