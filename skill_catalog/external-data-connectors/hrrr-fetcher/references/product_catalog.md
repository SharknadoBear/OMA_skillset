# HRRR products, grids, and sources

## Providers

The default anonymous HTTPS order is:

1. AWS NODD: `https://noaa-hrrr-bdp-pds.s3.amazonaws.com`
2. Google Cloud: `https://storage.googleapis.com/high-resolution-rapid-refresh`
3. Azure: `https://noaahrrr.blob.core.windows.net/hrrr`
4. NOMADS: `https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod`

AWS and Google contain the long archive beginning in 2014. Azure currently begins in 2021. NOMADS is a short-retention operational source. Runtime object probing is authoritative.

## Object naming

```text
hrrr.YYYYMMDD/{conus|alaska}/hrrr.tCCz.{wrfsfc|wrfprs|wrfnat|wrfsubh}fFF[.ak].grib2
```

CONUS omits `[.ak]`; Alaska requires `.ak`. Every GRIB object is accompanied by `.idx`. `CC` is initialization hour and `FF` is the enclosing forecast hour. Sounding files follow `hrrr.tCCz.class1.bufr[.ak].tm00`; v1 records their existence but does not decode BUFR.

## Native dimensions

| Domain | Projection | Grid | Analysis cycles |
|---|---|---:|---|
| CONUS | GRIB template 3.30, Lambert conformal | 1799 x 1059 | Hourly |
| Alaska | GRIB template 3.20, polar stereographic | 1299 x 919 | Every 3 hours |

`wrfprs` currently contains 39 isobaric levels from 50 through 1000 hPa plus auxiliary messages. `wrfnat` contains 50 hybrid levels plus surface/diagnostic messages. `wrfsubh` files `f01` through `f18` contain fields valid 15, 30, and 45 minutes before the hour and at the hour. BUFR sounding archives are intentionally outside v1 decoding.

As a current dimensional guide, `wrfsfc` has about 170 CONUS or 169 Alaska records, and every record is a two-dimensional field. Pressure/native messages add a vertical index through repeated two-dimensional GRIB messages; the canonical writer assembles those into explicit vertical dimensions. Counts are guidance only: live `.idx` and Section 3 metadata remain authoritative.

## Canonical aliases

| Alias | GRIB selection |
|---|---|
| `wind_10m` | paired UGRD/VGRD, 10 m above ground |
| `wind_80m` | paired UGRD/VGRD, 80 m above ground |
| `surface_pressure` | PRES, surface |
| `mean_sea_level_pressure` | MSLMA, mean sea level |
| `air_temperature_2m` | TMP, 2 m above ground |
| `specific_humidity_2m` | SPFH, 2 m above ground |
| `dew_point_temperature_2m` | DPT, 2 m above ground |
| `surface_temperature` | TMP, surface |
| `precipitation_rate` | PRATE, surface |
| `total_precipitation` | APCP, surface accumulation |
| `downward_shortwave_flux` | DSWRF, surface |
| `downward_longwave_flux` | DLWRF, surface |
| `sensible_heat_flux` | SHTFL, surface |
| `latent_heat_flux` | LHTFL, surface |
| `wind_gust` | GUST, surface |
| `visibility` | VIS, surface |
| `total_cloud_cover` | TCDC, entire atmosphere |
| `precipitable_water` | PWAT, entire atmosphere column |
| `composite_reflectivity` | REFC, entire atmosphere |
| `aerosol_optical_thickness` | AOTK, atmospheric column |
| `column_smoke_mass_density` | COLMD, atmospheric column |

Use exact selectors for pressure levels, hybrid levels, PMTF/PMTC smoke concentrations, or any field not represented by an alias. The `products` command prints the machine-readable alias catalog. Live `.idx` content and decoded ecCodes metadata override static record counts.

## Authoritative references

- [NOAA HRRR overview and data access](https://rapidrefresh.noaa.gov/hrrr/)
- [NCEP HRRR product inventories](https://www.nco.ncep.noaa.gov/pmb/products/hrrr/)
- [AWS Registry of Open Data: NOAA HRRR](https://registry.opendata.aws/noaa-hrrr-pds/)
- [Azure HRRR dataset layout](https://microsoft.github.io/AIforEarthDataSets/data/noaa-hrrr.html)
- [NOMADS data access](https://nomads.ncep.noaa.gov/)
