# GSHHG/GSHHS Sources

Use this reference when changing source handling for `gshhs-coastline`.

## Primary Dataset

The upstream dataset is commonly called GSHHG, while the shoreline shapefile
folder and files are named `GSHHS_shp` and `GSHHS_<resolution>_L<level>.shp`.
The skill name stays `gshhs-coastline` for consistency with the shapefile
artifact name.

Primary source URLs:

```text
https://ftp.soest.hawaii.edu/gshhg/gshhg-shp-2.3.7.zip
https://www.ngdc.noaa.gov/mgg/shorelines/shorelines.html
https://docs.generic-mapping-tools.org/6.2/datasets/gshhg.html
```

Use SOEST as the live ZIP source. Treat NCEI and GMT pages as source
documentation and provenance references. Cache first whenever possible.

## Resolution And Levels

GSHHG/GSHHS provides five coastline resolutions:

```text
c  crude
l  low
i  intermediate
h  high
f  full
```

The default workflow should use `h` for bpoly-scale regional topology and `f`
for small estuary or inlet detail. Level 1 is land and is the default for FVCOM
preprocessing topology. Additional levels can represent lakes and nested
features, but should be requested deliberately by downstream workflows.

## Caveats

GSHHG/GSHHS is globally consistent and polygonal, so it is a better base for
wet/dry topology than fragmented local line products. It is not a substitute
for high-accuracy local contemporary shoreline surveys. Use CUSP later as a
local detail/refinement overlay only after the GSHHS topology component is
known.

For FVCOM topology, distinguish the controlling model bbox from the source
clip. Center the model bbox inside a source footprint at least twice as wide
and high; use three times by default and enlarge symmetrically when the
RegionBPoly feedback look-ahead needs more room. Derive physical coastline
from the original polygon boundary before clipping. A clipped land polygon's
boundary includes the artificial request frame and is never shoreline evidence.

Some official source polygons can be invalid for strict GEOS overlay even when
the shapefile is readable. Validate only the features selected for a request,
repair invalid polygons in memory with `make_valid`, discard non-polygonal
repair remnants, and derive coastline from that validated polygon geometry.
Never rewrite the cached GSHHG files. Record the original validity reason,
repair method, and equal-area change so downstream users can audit the repair.

## Local Cache Policy

Search for cache in this order:

```text
<requested --cache-dir>/GSHHS_shp
<requested --cache-dir>
Workspace/Preprocessing/fvcom-gshhs-coastline/cache/gshhg/GSHHS_shp
Workspace/Preprocessing/fvcom-cusp-coastline/cache/gshhg/GSHHS_shp
```

The legacy CUSP preprocessing cache may already contain `h` and `f` resolution
level-1 shapefiles and should be reused rather than downloaded again.
