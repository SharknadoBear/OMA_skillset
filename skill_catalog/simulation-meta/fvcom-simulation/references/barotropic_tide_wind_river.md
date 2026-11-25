# Barotropic tide, wind and river discharge

Use `mode: barotropic-tide-wind-river` to combine a prescribed tide, atmospheric surface stress and pressure, offshore sea-level variability, boundary inverse-barometer response, and observed freshwater volume inflow. This is a three-dimensional barotropic experiment with inactive temperature and salinity dynamics. Apply the [wind forcing checks](barotropic_tide_wind.md) and the [river forcing checks](barotropic_tide_river.md) to the same staged namelist and runtime directory.

## Prepare the four Columbia members

The `columbia_wind_river_factorial` profile uses the retained Columbia mesh, 22 TPXO constituents, the existing station and section geometry, a 1.2 s step and the pressure-enabled FVCOM executable. Cold-start on March 18, 2025 UTC with a two-day ramp. Allow 14 days of spin-up and analyze April 1 through April 14. April 15 00:00 UTC is the production endpoint.

| Member | Mode | Prescribed ocean elevation | Surface forcing | River |
|---|---|---|---|---|
| Tide | `barotropic_tide` | TPXO | Constant 101325 Pa | Off |
| Wind | `barotropic-tide-wind` | TPXO + offshore anomaly + one inverse-barometer head | Variable stress and pressure | Off |
| River | `barotropic-tide-river` | Same as tide | Constant 101325 Pa | Port Westward |
| Combined | `barotropic-tide-wind-river` | Same as wind | Same as wind | Same as river |

The grid role verifies the retained mesh, ocean node order, 19 river receiving nodes, downstream section, upstream volume, and fixed station geometry. The tide role verifies the astronomical and composite boundary files and the surface product, then builds one river package shared by both river-bearing members. The configuration role verifies executable capabilities and renders both forcing blocks together. The lead freezes the scientific inputs and records the source, installation and artifact hashes.

Use HRRR, ERA5, CFSv2 and historical CFSR in that order for atmospheric forcing. Use GLORYS then HYCOM for offshore elevation according to the wind workflow. For a matched study use identical wind products on both wind-bearing members. Port Westward USGS site `14246900`, parameter `72137`, supplies tidally filtered hourly main-stem discharge. Fetch March 17 through April 16 inclusive, retain approval flags and original units, verify uninterrupted hourly coverage, and interpolate onto the six-minute integration clock. Distribute the unramped volume across the original width-and-depth-weighted cutoff nodes with uniform vertical weights. FVCOM applies the cold-start ramp during integration.

## Kestrel execution and reuse

Previously validated tide and wind members from `columbia_wind_pair` can be reused after verifying their native audits, input freezes, executable digest, clocks, mesh, gauge mapping and observation hashes. Preserve their original study and job identities. The older `columbia_river_pair` has a three-day spin-up and a different executable. Bind a new river production to the 28-day clock and the wind-study binary. Use one immutable study root and designate the two archived baselines by reference.

Run a 24-hour smoke on 52 ranks for each new member. Require the native river-reader count, atmospheric reader configuration, exact clocks, complete restart, and source-derived open-boundary audit, including the minimum-depth floor at intertidal nodes. Check native flags and positive water columns at every river receiving node. Verify a detectable seaward export response against each matching archived smoke. After both pass, run the new 28-day productions sequentially on two nodes with 208 ranks. Each production has 6,721 station records, 225 grid snapshots, and the April 15 restart. Keep individual attempts, Slurm receipts, and verified transfers in the study.

## Compare the four treatments

Run the tidal-validation specialist’s regional validator on each new production and its `factorial_validation.py` on the four audited attempts and frozen section geometry. Keep one shared six-minute observation support at each gauge. Report signed bias, raw and centered RMSE, residual correlation and peak timing. Retain prediction-only gauges. Document wetness at mapped gauge cells and availability of observed current profiles.

Calculate river minus tide, combined minus wind, wind minus tide and combined minus river. Their difference of differences is `combined - wind - river + tide`. Apply those contrasts to absolute elevations and to each depth-mean velocity component before deriving vector magnitudes. Use coincident wet locations and document the wet support. For both river increments, report ramped input, additional downstream section export, upstream storage change and residual over the full integration and analysis interval. The wind increment includes the complete atmospheric and offshore forcing package.
