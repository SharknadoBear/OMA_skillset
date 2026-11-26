# Validate a Lake Superior hindcast

Run `scripts/lake_validation.py` with Python, NumPy, pandas, netCDF4 and Matplotlib. The project root holds the frozen request, grid worker manifest, station mapping at `agents/01_grid/stations/station_mapping.json`, one archived NOAA station file at `agents/02_forcing/reference/*_stations_*.nc`, and the prepared, hash-bound observation tables in `analysis/observations`. The CLI reads the UTC analysis window and scheduled native output intervals from `request.json`.

## Hindcast

```shell
python scripts/lake_validation.py validate --project CASE --station STATIONS.nc --fields HISTORY_01.nc HISTORY_02.nc --model-datum lwd --out CASE/analysis/july_validation
```

Pass all station and history segments for the requested month. The native names, station order, sampled cell coordinates and bathymetry must match the station mapping. Where the startup and forcing preserve Lake Superior Low Water Datum, `--model-datum lwd` uses the documented conversion in the mapping to compare absolute modeled elevation with observed IGLD85 elevation. For Lake Superior this offset is 183.2 m. With an unknown model datum the CLI computes common-period elevation anomalies. [NOAA LSOFS model documentation, section 3.2](https://tidesandcurrents.noaa.gov/ofs/publications/CS_TechMemo_048_SA_LSOFS_2022.pdf)

For large native histories, copy both `scripts/lake_native_io.py` and `scripts/lake_validation.py` to the execution site. The reducer reads one upper-layer field at a time. It checks the native node order, weights temperature by native spherical nodal dual area and excludes every node marked dry. A wet node with missing temperature invalidates that record. Transfer the compact CSV and its sibling JSON receipt together.

```shell
python scripts/lake_validation.py compact --metrics native_horizontal_metrics.npz --fields HISTORY_01.nc HISTORY_02.nc --start 2025-07-01T00:00:00Z --end 2025-08-01T00:00:00Z --out lake_surface.csv
python scripts/lake_validation.py validate --project CASE --station STATIONS.nc --lake-series lake_surface.csv --model-datum lwd --out CASE/analysis/july_validation
```

The receipt binds the CSV to the grid worker’s area array and to the native sources. Validation requires every expected station timestamp and every daily GLSEA matchup. Use `--partial` for explicitly provisional output. Only days with a complete UTC sample clock enter the GLSEA score. The daily table retains each day’s sample count, missing and off-clock records, wet fraction, and observation availability. `--station-seconds` and `--grid-seconds` specify the actual planned sampling when it differs from the request. The output directory must be fresh.

## Compare an archived NOAA baseline

```shell
python scripts/lake_validation.py validate --project CASE --reference-only --reference NOAA_STATIONS.nc --fields NOAA_FIELDS.nc --partial --out CASE/analysis/noaa_reference_check
```

Archived mode checks the reference hash in the mapping and reports the observation sites available in that archive. The default reference station cadence is measured from its native clock and the default archived gridded cadence is hourly. A single midnight field provides a spatial check and a partial daily record. For a daily SST score supply all expected records for that UTC day. The July 2025 Lake Superior archive had 7440 six-minute station records covering 31 days.

The July 2025 NOAA station clock recorded 31 samples one second before the scheduled six-minute slot. The coverage audit assigns records within one second to the nominal slot, counts timestamp jitter and rejects duplicate slots. Physical interpolation retains the recorded UTC times.

The FVCOM_MP station file stores single-precision days and integer model steps. Where the stored days resolve only tens of seconds, the hindcast reader recovers the requested station cadence when every decoded time lies within its floating-point rounding interval and the internal step differences agree across the segment. The manifest records the number of adjusted timestamps, maximum shift and inferred seconds per internal step. For a complete diagnostic day inside a monthly project, use `--partial --analysis-start 2025-07-01T00:00:00Z --analysis-end 2025-07-02T00:00:00Z`. It reports coverage on this half-open window and retains the full project dates.

## Read the outputs

- `observation_metrics.csv` and `observation_pairs.csv.gz` give the independent gauge and buoy comparisons. Temperature is interpolated in time and between moving sigma midpoints using instantaneous local water depth. The surface and bottom half-layers use the nearest model layer. Measurements below the model bed are omitted. The metrics report the number of resolved pairs and samples using an outer half-layer. Surface-only observations with undocumented sensor depth use the upper layer.
- `thermocline_pairs.csv` uses the same available sensor depths on both profiles. The diagnosed depth is the midpoint of the strongest adjacent stable temperature gradient of at least 0.15 °C/m with three or more common depths. Single-sided detections have their own counts.
- `glsea_daily.csv` compares the area-weighted upper layer to daily GLSEA. This comparison includes differences in the skin-temperature product and the model layer’s vertical support.
- `model_reference_metrics.csv` in hindcast mode reports NOAA model-to-model elevation and temperature at coincident UTC times and equal sigma layers. The mapped FVCOM_MP cell shares the NOAA reference’s native node. Their horizontal footprints and physical depths may differ.
- `observed_skill.png` plots water-level residuals and temperature error by sensor depth. `validation_manifest.json` records source hashes, dataset roles, coverage, datum evidence and output hashes. A full-coverage result can still have substantial physical bias.

Run `python scripts/selftest_lake_validation.py` to check wet-area weighting, native coordinate ordering, sensor-depth interpolation, time support, datum handling, thermocline detection and the NOAA comparison branch.
