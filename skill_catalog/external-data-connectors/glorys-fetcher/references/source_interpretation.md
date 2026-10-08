# GLORYS12 source interpretation

The product is GLOBAL_MULTIYEAR_PHY_001_030, a global 1/12 degree physical reanalysis. Its published daily and monthly regular longitude/latitude fields are interpolated from the NEMO model grid. They are collocated; they must not be called the native Arakawa C grid. Plan against the actual published axes and catalogue, not approximate resolution arithmetic.

`zos` is sea-surface height above the source geoid in metres. `thetao` is sea-water potential temperature, normally degrees Celsius. `so` is source salinity, normally in `1e-3`; preserve its producer convention and attributes. `uo` and `vo` are eastward and northward horizontal velocities, normally m/s, at existing positive-down depth levels. `currents_3d` selects these depth-resolved horizontal components; it does not promise a vertical velocity field. The catalogue may also expose bottom temperature, mixed layer thickness and ice quantities; their regional applicability and masks differ.

Daily means average midnight-to-midnight and are described as centered at noon in the product manual. Stored source clocks govern selection. Catalogue minimum/maximum time metadata alone may have a different phase, so read the actual axis. Monthly fields are monthly means. Neither daily means nor a four-day pilot resolves all low-frequency variability; a longer period is needed for subsequent boundary experiments.

Do not add a guessed tide, vertical datum offset, land fill, below-seabed extrapolation or interpolation in the fetcher. Coupling non-tidal GLORYS sea level to tidal forcing requires a downstream decision about reference levels, overlapping atmospheric response and source variability. Source SSH is not automatically NAVD88 or a local gauge datum. For T/S and current forcing, use the downstream model's vertical coordinate and boundary mapping workflow after reviewing source coverage.

Health warnings use broad diagnostic limits: SSH ±10 m; potential temperature −4–45 °C; salinity 0–50 in source convention; horizontal velocity ±10 m/s. They are screening thresholds, not edits, and do not substitute for regional physical interpretation. Vector grids must align; any U/V mask differences are counted and retained.

Primary references:

- [GLORYS product manual](https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-030.pdf)
- [Toolbox subset documentation](https://toolbox-docs.marine.copernicus.eu/en/stable/usage/subset-usage.html)
- [Toolbox Python interface](https://toolbox-docs.marine.copernicus.eu/en/stable/python-interface.html)
- [Toolbox local login](https://toolbox-docs.marine.copernicus.eu/en/stable/usage/login-usage.html)
- [Marine registration and CDSE guidance](https://help.marine.copernicus.eu/en/articles/4220332-how-to-register-for-copernicus-marine-service)
