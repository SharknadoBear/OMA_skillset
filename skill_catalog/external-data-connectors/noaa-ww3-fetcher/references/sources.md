# Source and decoding notes

- Archive: https://polar.ncep.noaa.gov/waves/hindcasts/multi_1/
- README: https://polar.ncep.noaa.gov/waves/hindcasts/multi_1/README.txt
- Definitions: https://polar.ncep.noaa.gov/waves/products.shtml
- WW3 source: https://github.com/NOAA-EMC/WW3 (ww3_grib.F90, ww3_outp.F90)

This production hindcast uses GFS analysis winds. Discover monthly availability rather than extending the README's stale endpoint. Fields are `YYYYMM/gribs/multi_1.GRID.FIELD.YYYYMM.grb2`; points are `YYYYMM/points/multi_1_base.buoys_spec.YYYYMM.tar.gz` and `buoys_wmo`. Adjacent `.MD5` files are publisher checksums. HEAD sizes refer to compressed wire bytes. Some tar.gz responses carry Content-Encoding gzip; requests' normal iter_content expands them and invalidates checksums. Read raw with decode_content=False.

Fields normally have three-hour valid times. Installments may repeat a segment boundary and include next month's first record. Verify regional duplicate values before deduplication. Preserve source band, reference time, forecast seconds and GRIB2 packing increments per output time. Circular direction values can slightly cross 0/360 through packing; retain them and allow half the native packing increment plus float32 rounding in health checks. Preserve GRIB comment “Primary wave mean period” while interpreting PERPW as peak period from NOAA's product table. Direction documentation varies; preserve native labels and avoid direction-dependent inference without additional confirmation.

ASCII SPEC starts with quoted header, nf, nd, npoints; frequency list; direction list; date/time; quoted point header with latitude, longitude, depth, winds and currents; then nf*nd values. Frequency index varies fastest. Directions are Cartesian travel directions in radians; preserve order and wrap. Energy density is per Hz per radian (historical README omits angle denominator); verify through WMO Hs.

Integration estimates native geometric ratio XFR from a log-linear fit to the whole printed frequency axis (not the old example 1.1). Three-significant-figure axes have uneven adjacent ratios; reject deviations exceeding 0.6% from the fitted axis. Preserve printed frequencies and use bandwidth `0.5*(XFR-1/XFR)*f` and periodic direction widths. Hs = 4 sqrt(sum E df dtheta). QA tolerance max(0.1 m, 5% of rounded WMO Hs) allows ASCII precision, one-decimal WMO rounding and unresolved high-frequency tail conventions. Record errors; never normalize to force agreement. WMO rows: year, month, day, hour, wind speed/direction, Hs, Tp. Measure cadences independently.

Version 1 handles native regular geographic GRIB grids and non-dateline bounds. It rejects rotated/irregular grids, wrapped crops and implicit grid-changing concatenation. No spectra are spatially interpolated.
