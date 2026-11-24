# Offshore forcing for barotropic-tide-wind

Generate the matched control with `fvcom-tpxo-tides`. Retain its unmodified NetCDF, owner manifest and ordered geographic point CSV. Use `glorys-fetcher` to acquire padded P1D `zos` over the run and all 240 P1M fields from January 1993 through December 2012. Both acquisitions must pass provider health. The atmospheric builder must supply the same absolute hourly nodal pressure field used by FVCOM.

The supplied GLORYS12v1 solution omits astronomical tides and the inverse-barometer response. This lets the boundary include both independently. Regional wind-driven ocean variability may already occur in the offshore anomaly. [GLORYS producer description](https://www.mercator-ocean.eu/wp-content/uploads/products_pdf/MOI_GLORYS12.pdf) · [Source convention evaluated in section 2](https://doi.org/10.3390/jmse13122379)

Use the same bilinear stencil for daily and historical SSH. Where a stencil includes land, the builder uses the nearest wet grid point shared by all source records within an explicit regional cutoff. Review the recorded distances and coastal geometry. Use the [documented HYCOM fallback](hycom_reference.md) when GLORYS is unavailable.

The daily archives can label a day's mean at 00 UTC. The downstream reconstruction records this native UTC clock and locates the documented midnight-to-midnight daily mean at its noon midpoint. It removes one constant per node, obtained by weighting the 240 monthly means by their actual calendar-day counts, including five leap days. It then interpolates the resulting anomaly to the six-minute tide clock between source midpoints. [Copernicus Marine product manual, section III](https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-030.pdf)

For the pinned Cartesian, type-1 FVCOM time-series boundary, the prescribed sea level is

```text
elevation = TPXO + (daily_zos - reference_zos) - (pressure_pa - 101325) / (1025 * 9.81)
```

The reference uses the local 1993–2012 model mean above geoid. Absolute conversion to a survey or gauge datum requires an independent vertical reference. Positive pressure departures lower the boundary. The type-1 clamp in `mod_obcs.F` applies the supplied physical sea level. The Cartesian `GRAV_N` in `mod_setup.F`, `GRAV` and `ELF_AIR` in `mod_main.F`, and the atmospheric pressure gradient in `advave_edge_gcn.F` govern the distinct interior momentum term. Use this policy only with a compatible inspected build.

```powershell
python scripts/wind_boundary.py --tide control/columbia_elevtide.nc --tide-manifest control/forcing_manifest.json --points ordered_obc.csv --daily offshore/daily --monthly offshore/monthly --mesh columbia_grd.dat --obc columbia_obc.dat --pressure columbia_surface.nc --nearest-limit-km 12 --output-dir treatment
python scripts/selftest_wind_boundary.py
```

Stage the four treatment outputs together with the original mesh, OBC DAT, atmospheric file and adjacent `atmosphere_manifest.json` from the surface producer. Stage `forcing_manifest.json` from the unchanged pure tide separately for the normal tidal checks. Each treatment verifier calls `verify_wind_boundary(input_root, namelist_values)` before running or scoring. It checks artifact hashes and independently reconstructs the reference, offshore anomaly, pressure response and complete boundary matrix using native SSH samples and the staged atmospheric pressure. The returned path identifies the original tide suitable for the existing TPXO checks. The control uses the original tide directly.
