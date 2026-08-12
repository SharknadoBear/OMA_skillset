# NOAA EFDC map source contract

SJROFS native fields are structured curvilinear EFDC aggregates with dimensions approximately `time,sigma,ny,nx`. Coordinates and dynamic variables are collocated. Native `u` and `v` must declare `eastward_sea_water_velocity` and `northward_sea_water_velocity`; no C-grid destaggering or vector rotation applies.

The source `mask` is authoritative: exactly code `5` is active water. Code `0` and the known negative padding sentinel are inactive. `wet_mask` in an `efdc_compact_fields_v1` file is derived binary metadata and must exactly match source `mask == 5`. Finite hydrodynamic values outside those cells or an unfamiliar positive mask code are corruption/contract failures. Atmospheric source components may validly cover dry packed cells; loaders accept that source coverage but mask wind scalar/quiver results back to active water.

The source sigma values are positive-down layer-top fractions. They must be unique in `[0,1)`, with the minimum within `1e-6` of zero. Sort by value, append the bed edge `1`, take successive differences, then return weights to source storage order. The method identifier is `efdc_layer_top_sigma_with_bed_edge_1`.

Default plots use independent wet-cell polygons. Footprint length and orientation are inferred only through immediate wet logical neighbors. Dry/padding coordinates are never read for this calculation. Center-coordinate `pcolormesh` is rejected on masked inputs.
