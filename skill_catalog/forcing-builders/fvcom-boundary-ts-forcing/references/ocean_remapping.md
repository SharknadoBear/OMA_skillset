# Ocean profiles at prescribed FVCOM elevation

The optional remapper targets one or more ordered geographic OBC nodestrings. Supply the same runtime depth, OBC list, uniform sigma definition and time-series elevation used in the native run. Set the minimum wet depth from the run namelist. Specify a reviewed offshore seed that stays wet to the deepest target and a regional limit on donor distance. Review the method counts and distances before acceptance.

## Ocean products

For daily GLORYS12, fetch `so` and `thetao` together with `glorys-fetcher`. Keep its original `subset.nc`, `manifest.json` and `health_report.json` in the source directory. Both fields share the same finite donor stencil through the full source period and both endpoints of each vertical bracket. The remapper linearly interpolates at the provider's recorded UTC labels. The higher output cadence resolves changing FVCOM water depth. It does not add subdaily ocean analyses.

For an approved salinity-only fallback, use `hycom-fetcher` to inventory and subset `salinity(time,depth,lat,lon)` from the annual three-hourly ESPC-D-V02 s3z analysis. Keep the hash-bound download plan, `health_check.json`, and its named subset in the same directory. Pass `--source-provider hycom-espc` and the single-record `--frozen-temperature-startup`. The separate t3z temperature product is unnecessary for a fixed-temperature experiment. This fallback handles packed salinity, the 0–360 longitude convention, the 0 m surface level, and three-hourly instantaneous values. A prognostic temperature run requires an independently qualified joint T/S product. [HYCOM analysis description](https://www.hycom.org/dataserver/espc-d-v02/global-analysis) [2025 salinity catalogue](https://ncss.hycom.org/thredds/catalogs/ESPC-D-V02_s3z.html?dataset=ESPC-D-V02-s3z-2025)

An example HYCOM connector request for the Columbia shelf follows. Inventory the URL, estimate and fetch according to the connector before remapping.

```json
{
  "source": "https://tds.hycom.org/thredds/dodsC/ESPC-D-V02/s3z/2025",
  "variables": ["salinity"],
  "start": "2025-03-17T15:00:00Z",
  "end": "2025-03-19T09:00:00Z",
  "bbox": [-124.72, 45.75, -123.72, 46.70],
  "depth": [0, 150],
  "output": "subset.nc"
}
```

## Remap and write

```powershell
python scripts/remap_glorys_sigma.py --source-provider glorys --source source_directory `
  --mesh-2dm estuary.2dm --open-ns 1 --obc estuary_obc.dat --dep estuary_dep.dat `
  --sigma estuary_sig.dat --elevation estuary_elevtide.nc --seed -124.5 46.25 `
  --max-donor-km 20 --min-depth 0.05 --start 2025-03-18T00:00:00Z `
  --end 2025-03-19T00:00:00Z --guard-hours 6 --step-seconds 360 `
  --frozen-temperature-startup estuary_start.nc --output wet_remap
python scripts/build_fvcom_boundary_ts.py --source wet_remap/sigma_ready.nc `
  --mesh-2dm estuary.2dm --open-ns 1 --case-name estuary --output estuary_tsobc.nc
```

Set the frozen temperature option only for a native run whose temperature field remains fixed. GLORYS temperature remains an additional independently recoverable field in the sigma-ready product even when the native temperature companion uses the startup. HYCOM salinity-only outputs contain the startup companion and salinity.

The target layer depths in metres below the instantaneous surface equal `-siglay*(h+elevation)`. Donors must be four-connected through water to the offshore seed. The search uses bilinear weights, then renormalized wet-corner weights or the nearest connected wet center. Targets above the shallowest source level receive the surface value. A target below the source depth coverage fails. Dry prescribed nodes use a recorded reference-depth profile. Missing elevation guard records may also use reference bathymetry, while the entire requested experiment must have exact elevation coverage. Verify evolving model wetness and actual native boundary values after execution.

The immutable delivery contains `sigma_ready.nc`, `remap_manifest.json` and `remap_support.npz`. The latter preserves source and destination clocks, physical depths, land masks, connected components, donor indices, interpolation weights and distances. The standard writer adds the final classic NetCDF and its required QA figures.
