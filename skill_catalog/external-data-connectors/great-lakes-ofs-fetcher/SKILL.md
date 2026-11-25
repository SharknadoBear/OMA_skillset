---
name: great-lakes-ofs-fetcher
description: Discover, verify, resume and retrieve NOAA Great Lakes Operational Forecast System native FVCOM nowcast and forecast fields and station records for Superior, Michigan–Huron, Erie and Ontario. Use for archived three-dimensional water-column analyses, immutable same-time source selection or deduplicated model station references.
---

# Great Lakes OFS archive

Use Python 3.11+ with requests, h5py, NumPy and netCDF4. The public NOAA NOS OFS S3 bucket supports anonymous listing and version-bound byte ranges. Supply an explicit UTC time and one of `lsofs`, `lmhofs`, `leofs`, or `loofs`. Select a persistent nonsynced cache with space for the published object sizes.

## Find a snapshot

```text
python scripts/great_lakes_ofs.py inventory --system lsofs --time 2025-07-01T00:00:00Z --product fields --output inventory.json
python scripts/great_lakes_ofs.py metadata --system lsofs --time 2025-07-01T00:00:00Z --product fields --output metadata.json
python scripts/great_lakes_ofs.py fetch --system lsofs --time 2025-07-01T00:00:00Z --product fields --cache RAW_CACHE --output snapshot.json
```

Inspect published byte counts, object keys, native variables, dimensions, and internal times. The metadata reader requests HDF5 headers and time coordinates in small ranges. A full fetch builds a version-bound local object, checks its NetCDF clock, and returns the exact record number with object URL, size, ETag and SHA-256. Include `--existing PATH` to verify a previously downloaded source against the listed object. Use `--product stations` to obtain a native station cycle.

Prefer the archive nowcast, then the closest cycle, then the latest initialization at a tie. For forecast-eligible work explicitly add `--allow-forecast` and report the selected stream and lead. The reader confirms valid time inside the file. Historical archives may have gaps. Forecast eligibility, candidate bounds and gridded versus station clocks must be checked per request. A source snapshot contains native layers, staggered velocities, and published grid coordinates. Preserve these with the field values.

## Assemble station references

```text
python scripts/station_series.py --system lsofs --start 2025-07-01T00:00:00Z --end 2025-08-01T00:00:00Z --cache RAW_CACHE --output lsofs_july_stations.nc
python scripts/test_fetcher.py
```

The station window includes the start and excludes the end. Use `--cadence-seconds` for the native cadence and `--workers` to bound concurrent cache reads. Overlapping nowcasts use the same precedence rule. Verify native station geometry and IDs across cycles. The output preserves actual time values and masks. Its neighboring JSON records each contributing object, overlap count, native time deviations up to one second, final axes, and checksum. Keep model station series labeled as model references when comparing with independent instruments.

The archive root and system descriptions are available from the [NOAA NOS OFS Open Data Registry](https://registry.opendata.aws/noaa-nos-ofs/) and [NOAA Great Lakes operational forecasts](https://tidesandcurrents.noaa.gov/ofs/glofs.html).
