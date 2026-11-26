# Complete lake atmosphere and freshwater forcing

Run the selector in an environment with netCDF4, NumPy, SciPy and pyproj. Both UTC endpoints are inclusive whole hours. The geometry file must contain native FVCOM node and triangle coordinates, connectivity and nodal control-volume areas `art1`. Supply the frozen source file that defines `PSIMPLE_DRAG`.

```powershell
python scripts/ranked_lake_surface.py --mesh native_reference.nc --mod-bulk mod_bulk.F --start 2025-07-01T00:00:00Z --end 2025-07-02T00:00:00Z --source-root sources --output surface
```

The selector tries HRRR, ERA5, CFSv2 and CFSR in order. Each candidate must supply the wind, mean sea-level pressure, heat and freshwater fields and pass native generation, conservation and file checks. Inspect `provider_attempts.json` for individual failures and the selected manifest hash. The output contains `superior_atmosphere.nc`, `superior_pe.nc`, `surface_budget.csv` and `surface_manifest.json`. Use a new output directory for a different provider, period or geometry.

By default the selector reads checked caches. Add `--acquire --bbox WEST SOUTH EAST NORTH` to invoke the installed archive skills for missing data. Supply `--hrrr-python`, `--era5-python` and `--cfs-python` when their pinned decoder or CDS environments differ from the model environment. Set `--skill-root` to the directory containing the four connector packages if running from a development checkout. Production NCEI fetches can require full monthly GRIB objects. Inspect the connector's plan estimate for the requested area, dates and products before transferring large runs.

| Provider | Checked source cache | Atmospheric and interval inputs |
| --- | --- | --- |
| HRRR | `sources/hrrr/YYYYMMDDHH` and `sources/hrrr_precip/YYYYMMDDHH` for six-hour UTC batches, or `--hrrr-analysis` and `--hrrr-precipitation` | Native CONUS f00 paired winds, MSLP and instantaneous down/up solar and thermal radiation and turbulent heat fluxes. Each separate f01 APCP accumulation covers its forecast cycle to cycle + 1 h. |
| ERA5 | `sources/era5` from the ERA5 fetcher, with `wind_10m`, `mean_sea_level_pressure` and `surface_heat_and_precipitation` | Hourly geographical wind and absolute MSLP. The CDS HRES `ssr`, `str`, `sshf` and `slhf` are signed downward energy integrals in J/m² and `tp` is liquid-water depth in m. A record at midnight supplies the hour ending at midnight. |
| CFSv2 / CFSR | `sources/cfsv2` or `sources/cfsr`, with a native NCEI whole-request lock and health receipt | Native 10-m winds, PRMSL, down/up solar and thermal flux, upward sensible and latent heat flux, and precipitation rate. The era transition is 2011-04-01 00 UTC. A single provider must cover the complete window. |

For CFS, native one-hour means pass through directly. For two cumulative means spanning the same forecast cycle and lower bound, the last hour's integral equals the difference of mean times duration. The connector retains the positive-duration ending mean at a cycle boundary that also contains a zero-duration analysis. Missing prefixes, inconsistent clocks and unsupported intervals reject the provider. The pressure field can have a coarser native grid. It uses its own coordinate axes during interpolation.

For accumulated and interval-mean providers, the writer reconstructs a piecewise-linear time series on the hour, one minute past, one minute before the next hour and the next hour. Each node's stored trapezoidal integral recovers the source-derived hourly mean. Hourly pressure and stress interpolate linearly between endpoints. Earth-relative winds use east/north on spherical meshes. Supply `--projected-crs` only for a projected Cartesian solver.

The energy ledger uses a summer liquid-water latent heat of 2.5 MJ/kg and density of 1,000 kg/m³. Condensation adds to freshwater gains, evaporation is negative in the forcing file, and the hourly ledger lists rain, condensation and evaporative losses separately. An interval differenced from quantized CFS rates can yield small negative rain. The writer sets it to zero only within the propagated native GRIB packing error plus float32 storage precision. The CFS ledger includes the original integrated rain, the added volume and the number of affected model nodes for each interval. Larger deficits reject the candidate.

Run `python scripts/selftest_lake_intervals.py` and `python scripts/selftest_lake_surface.py` to check midnight alignment, signed heat and water, checksum failure, CFS de-averaging, precision limits, source choice and native readback.
