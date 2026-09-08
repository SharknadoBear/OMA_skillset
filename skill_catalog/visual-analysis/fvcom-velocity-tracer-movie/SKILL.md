---
name: fvcom-velocity-tracer-movie
description: Build standalone offline HTML visualizations of FVCOM ocean currents with native-mesh speed shading and seeded particles with fading tails. Use for snapshot time-slider exploration, experimental continuous current playback, presentation GIF export from a fixed snapshot, and reproducible visual QA of depth-averaged or available sigma-layer velocities.
---

# FVCOM Velocity Tracer Movie

Create one self-contained HTML per selected mesh/time interval. The default mode animates particles through a fixed native timestamp selected by a slider. Continuous mode experimentally interpolates the field on its model clock; Visual speed independently scales particle travel. Both modes are available in each HTML.

## Workflow

1. Inspect native output with the requested interval and layer. Resolve the coordinate CRS and velocity basis from actual metadata or run provenance. FVCOM files can contain zero-filled `lon/lat` despite valid projected `x/y`; do not plot those geographic placeholders.
2. Build a bounded HTML with native connectivity and wet masks. Use one explicit color limit for comparisons; otherwise the exporter uses the maximum selected finite speed. The end timestamp is inclusive and both requested endpoints must exist.
3. Open the HTML directly in a browser. Check the inlet/channel view as well as the full domain, direction changes across native timestamps, and the experimental transition between outputs. Keep the HTML and its JSON report together for provenance; only the HTML is needed to view it.
4. When iterating, use a new output directory and retain settings, screenshots, numerical checks, and browser performance. Install an exact validated catalog copy; generated datasets and figures belong outside this skill.

## Commands

Python dependencies: `numpy`, `netCDF4`, and `pyproj`. The viewer has no network or server dependency. WebGL2 supplies shading and finite-age instanced trails; Canvas trails and the map remain available without it.

```powershell
python scripts/fvcom_velocity_tracer_movie.py inspect `
  --input galveston_0001.nc --crs EPSG:32615 --vector-basis grid `
  --start 2025-04-01T00:00:00Z --end 2025-04-02T00:00:00Z `
  --output inspection.json

python scripts/fvcom_velocity_tracer_movie.py build `
  --input galveston_0001.nc --crs EPSG:32615 --vector-basis grid `
  --obc galveston_obc.dat --layer depth_average `
  --start 2025-04-01T00:00:00Z --end 2025-04-02T00:00:00Z `
  --vmax 1.6 --title "Galveston Bay currents" --output currents.html
```

Repeat `--input` or provide multiple paths after it. `--layer` accepts `depth_average` (native `ua/va`), `surface`, `bottom`, or `index:N` (native `u/v`). Do not label depth-averaged vectors as surface velocity. No wind or derived depth-average from 3D velocities is included.

`--vector-basis east-north` preserves native values and transforms motion components into the display projection. `--crs` is the CRS of projected source `x/y`; valid geographic coordinates can instead use a locally derived metric projection with earth-relative vectors. An OBC file must use the same node IDs as the NetCDF. Without it, boundaries remain explicitly unclassified.

Read [scientific_methods.md](references/scientific_methods.md) for interpolation, boundary handling, temporal gaps, and interpretation. Read [output_contract.md](references/output_contract.md) when integrating the Python exporter or diagnosing a report. Earth attribution is bundled in the viewer and documented in [earth_provenance.json](references/earth_provenance.json).

## Snapshot GIF export

In the HTML, select Snapshot, choose a native timestamp and map view, adjust Appearance, then use **Export GIF**. Defaults are five seconds, 20 FPS, and 2400 pixels wide with the visible map aspect ratio. This is eight inches at 300 pixels/inch; GIF has no physical DPI metadata. Only the fixed field is exported. The GIF includes scientific labels and omits controls and hover information.

The exporter warms reproducibly seeded tails, renders at fixed 60 Hz, and encodes in an offline worker. It preserves logical map scale, boundary checks, and finite trail expiry. Width choices are 1200/1800/2400 pixels, duration 3-10 whole seconds, and frame rate 10/20 FPS. A visible forward-loop restart is expected. Download the optional settings JSON when retaining provenance. Continuous GIF export is unsupported.

Read [gif_export.md](references/gif_export.md) for runtime automation, limits, encoding, and validation. Generated GIFs, HTML products, and review evidence belong outside the skill repository.

## Validation

Run `python scripts/selftest_velocity_tracer.py` and skill-creator `quick_validate.py`. Run `scripts/browser_numerics.js`, `scripts/browser_trails.js`, `scripts/browser_controls.js`, and `scripts/browser_gif.js` in a browser with the viewer loaded. They cover scaled RK2, barriers, wet sectors, unscaled time interpolation, exact trail expiry, history capacity, and independent controls. Each expression returns JSON and throws on failure.

The continuous Visual speed slider independently scales particle displacement; Playback duration controls model time. At 1× they retain the model-velocity relationship. Other gains are explicitly illustrative; never scale the legend, source data, or hover velocities. Snapshot and continuous settings are remembered separately. Defaults are 3 s and 1 s maximum trail duration respectively, 1× visual speed, 3,000 particles, and 60 s playback. Trail age freezes while paused or hidden and reaches exactly zero at expiry. Record fallback performance separately from the primary WebGL2 target.

Use a browser automation environment to check direct-file opening with networking disabled, native and interpolated timestamps, reset/pause controls, zoom/pan, shading fallback (`?no-webgl=1`), and a full continuous loop. Compare CPU/browser samples and shader colors against the Python reference, and record the graphics backend with measured FPS. Numerical correctness is required in both modes; continuous visual polish remains experimental.
