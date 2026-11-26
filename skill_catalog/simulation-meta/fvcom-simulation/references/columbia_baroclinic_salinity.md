# Estuarine salinity with a fixed temperature field

Select `baroclinic-salinity-tide-wind-river` for a marine baroclinic simulation driven by tides, wind stress, atmospheric pressure, offshore meteorological elevation, ocean salinity and observed river discharge. The `columbia_baroclinic_salinity_pilot` profile uses March 18 00:00 through March 19 00:00 UTC, 2025, with the retained Columbia mesh, nine uniform layers and 52 ranks on Perlmutter CPU. Choose a new scratch case directory and supply the allocation in the private run request.

## Preparation

1. Freeze the metric grid, depth, node/element ordering, sigma coordinates and ordered open boundary. Coordinate the initialization, offshore and river forcing, and native build in parallel. Preserve input and executable hashes in the stage handoff.
2. Use `fvcom-initialization` with a dated, wet-connected 3-D estuarine analysis for salinity, water level and currents. Freeze the dated analysis temperature at each location and layer. If temperature is unavailable, prefer a documented time-invariant spatial field such as a seasonal climatology, then a caller-supplied scalar. Record the tier used. Calculate seawater density with the resulting temperature and evolving salinity. Preserve the source and target vertical references, target-layer overlap, donor extrapolation and the hydrographic adjustment at the outer boundary. SSCOFS native `Times` determines the initialization instant.
3. Reuse the checked astronomical tide, meteorological sea-level anomaly, wind stress, atmospheric pressure and single inverse-barometer correction. Use `glorys-fetcher` for global salinity, then `hycom-fetcher` for an alternative complete interval if the primary lacks coverage. Use source valid times, depth support, wet-mask connectivity and true practical salinity units when producing layer profiles for `fvcom-boundary-ts-forcing`. On this FVCOM revision the inflow relaxation coefficient is dimensionless per internal step and 1 gives the prescribed inflow profile. Ebb uses the native outgoing tracer treatment. Retain discharge magnitudes, river distribution and timestamps. Set the Columbia freshwater endmember to 0 practical salinity with discharge-weighted tracer input.
4. Use `fvcom-namelist-configuration`, `fvcom-build` and `fvcom-run-control` for the marine executable, dated startup, salt and freshwater budget diagnostics, and two immutable jobs. The short reader must exercise real run inputs. A new forecast initialized from the analysis uses full-strength forcing at startup. Hold temperature transport and surface heat flux inactive. Check temperature and density in native fields, the initial and imposed ocean elevation and the runtime salinity reader. Then use one 24-hour job at 1.2 s external timestep. Verify exit status, the completion marker, station/grid clocks including the initial epoch, and a readable 3-D restart ending at the requested instant.

For newly prepared atmospheric inputs, use HRRR first, followed by ERA5, CFSv2 and CFS in coverage order. Use the existing [wind and meteorological boundary workflow](barotropic_tide_wind.md) for source pressure, stress and sea-level conventions. Choose the first complete ocean salinity source independently of the elevation source so a salinity fallback preserves a verified meteorological tide.

## Validation

Archive direct gauge salinity with provider identity, QC definitions, sensor elevation and vertical datum. For conductivity-derived salinity, record the conductivity convention and use the measured water temperature. Sample genuine wet model profiles at the instrument elevation, using a documented local datum conversion and the actual observation timestamps. A single site with multiple depths measures local stratification and passage. An observed channel intrusion distance requires spatially bracketing gauges.

Use the validation entry point after the native execution audit. It checks hashes from the closed run and the acquisition receipts, checks post-initialization hourly coverage, then writes gauge comparisons, channel sections, front distances and salt budgets together. A successful validation receipt records the output hashes. Keep scientific interpretation in the accompanying case report.

```text
python scripts/columbia_channels.py --grid retained_columbia_grid_output.nc --output transects
python scripts/estuary_validation.py --attempt returned_attempt --ofs-manifest sscofs/fetch_manifest.json --channels transects/channels.geojson --sensors sensors.json --obc columbia_obc.dat --start 2025-03-18T00:00:00Z --end 2025-03-19T00:00:00Z --internal-step-seconds 1.2 --output comparison
```

The component scripts below also run independently. The section input is a GeoJSON FeatureCollection of mouth-to-upstream LineStrings, each with a `channel` name and matching `distance_km` array. Freeze representative wet thalwegs independently of salinity. Sample both meshes at 1 m above their respective beds, cap the end-layer value only within the physical column, and retain dry/off-mesh gaps. A connected front ends at the first upstream threshold crossing. Mark a gap or the end of a salty transect as a lower bound.

```text
python scripts/estuary_sections.py --files fvcom_0001.nc --crs EPSG:32610 --channels channels.geojson --output model_sections
python scripts/estuary_sections.py --files [SSCOFS field files] --channels channels.geojson --output ofs_sections
python scripts/estuary_comparison.py --model model_sections --reference ofs_sections --start 2025-03-18T00:00 --end 2025-03-19T00:00 --output comparison
python scripts/estuary_gauges.py --stations columbia_station_timeseries.nc --startup-utc 2025-03-18T00:00:00Z --internal-step-seconds 1.2 --sensors sensors.json --output comparison
python scripts/estuary_budget.py --grid fvcom_0001.nc --obc columbia_obc.dat --ledgers returned_attempt --ranks 52 --internal-step-seconds 1.2 --output comparison
```

The gauge specification is an array. Each item supplies `id`, `station` (native station name), `csv` (ERDDAP CSV with UTC time, salinity, salinity_qc and units row), `accepted_flags` (strings), `elevation_in_model_datum_m` (positive up), `datum_evidence` and `qc_evidence`. Compare paired bias, RMSE, centered RMSE, correlation and deep-minus-shallow stratification. Score OFS against FVCOM at the 24 subsequent hourly times. Report front distance, along-channel excursion and timing, spatial coverage, wet-layer salinity extrema and budget closure separately from execution status. Disclose common initialization, datum or endpoint limitations, and any residual that prevents an interpretation of conservation.

For salinity storage use wet control volumes and the same free surface as the native tracer step. Compare the storage change with measured freshwater input and gross ocean salt exchange. The budget CLI distinguishes the 3-D open-edge estimate from the native depth-mean transport estimate and the exact sum of numerical operator changes. Native budget closure also includes boundary replacement, tracer limiting and wet/dry effects. Practical salinity multiplied by volume has units of PSU m³.

The March window has two usable fixed intakes at SATURN-03. The provider’s nominal depths of 240 and 820 cm use NGVD29. At the intake, NGS VERTCON 3.0 converts zero NGVD29 elevation to +1.098 m NAVD88. The model station uses a three-vertex average in containing element 19424, whose centroid is 30.8 m from the intake. Preserve the bathymetry reference and check sensitivity to the tide series’ mean-level convention. [Upper intake and QC definitions](https://critfcweb.ohsu.edu/erddap/info/saturn03_240_A_CT/index.html) · [Lower intake](https://critfcweb.ohsu.edu/erddap/info/saturn03_820_A_CT/index.html) · [NGS transformations](https://www.ngs.noaa.gov/web_services/ncat/index.shtml)

Regional model and global boundary products are published separately. [NOAA SSCOFS](https://www.co-ops.nos.noaa.gov/ofs/sscofs/sscofs_info.html) · [Copernicus GLORYS12](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description) · [HYCOM 2025 archive](https://ncss.hycom.org/thredds/catalogs/ESPC-D-V02_s3z.html?dataset=ESPC-D-V02-s3z-2025)

## March pilot qualification

The original sloping-bed tracer correction produced a wet minimum of −0.02554 PSU. All 24 post-start hourly fields contained salinity below −0.001 PSU. The largest affected fraction was 0.00051% of wet volume. A matched reader isolated the undershoot to the optional bottom-slope term after the scalar positivity step.

To reproduce the nonnegative Columbia variant, select `bottom-sensitivity` for its reader and `bottom-sensitivity-day` for its full day in both the namelist and run-control CLIs. The day must use its matching reader. The two 24-hour jobs used the same executable and identical runtime inputs. Setting `NOFLUX_BOT_CONDITION=F` omits the slope term and retains implicit vertical mixing. All 25 saved wet fields then lay between 0 and 33.285 PSU. Retain the ordinary `reader` and `smoke` route to reproduce the original formulation, and reassess the bottom condition on other meshes.

| Comparison | Original | Bottom-slope term disabled |
|---|---:|---:|
| SATURN-03 upper intake RMSE / centered RMSE (PSU) | 2.96 / 2.06 | 2.38 / 1.83 |
| SATURN-03 lower intake RMSE / centered RMSE (PSU) | 5.64 / 2.64 | 4.89 / 2.61 |
| North Channel 1 PSU front RMSE against SSCOFS (km) | 4.68 | 4.10 |
| South Channel 1 PSU front RMSE against SSCOFS (km) | 6.54 | 4.46 |

Both runs captured the observed variability and underestimated mean salinity. In the nonnegative run the upper and lower biases were −1.52 and −4.14 PSU. The lower intake had 240 matched measurements and correlation 0.960. The 1 PSU front averaged 0.30 km farther upstream than SSCOFS in the North Channel and 2.55 km farther seaward in the South Channel. The net physical salt-transport proxy left a residual equal to −0.756% of gross exchange, or 60.5% of the net storage change. These figures describe this one-day diagnostic.
