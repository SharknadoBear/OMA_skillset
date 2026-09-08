# Exporter and HTML contract

`fvcom_velocity_tracer_movie.py` provides `prepare(inputs, start, end, layer, crs, vector_basis, obc, max_gap_hours)`, `reconstruct(mesh, vectors, valid)`, and `write_html(arrays, info, output, title, vmax)`.

- `prepare` returns NumPy arrays and JSON-compatible metadata. All selected records remain at native time cadence; interpolation is performed in the browser.
- `reconstruct` is the independent Python reference for connected-wet-fan corner vectors, returning `(element, corner, component)` Float32 data.
- `write_html` embeds the viewer and arrays, returning a manifest containing the output path, byte count, and SHA-256. The CLI writes this to a companion JSON, or `--report`.

Metadata uses `schema_version: fvcom_velocity_tracer_v1`. It records source paths, sizes and SHA-256 hashes, timestamps in UTC and epoch seconds, selected layer, units, CRS and original vector basis, coordinate recovery, native geometry counts, masks/coverage, excluded invalid vectors, temporal gaps, OBC identity, reconstruction method, source Earth revision, and display settings.

Embedded arrays are little-endian binary encoded as base64. Descriptors record dtype, shape, byte count, and decoded SHA-256. Coordinates `xy` and `geo` are Float64. Native triangle IDs and derived `neighbors` are Int32. Element area is Float64, minimum altitude Float32, and boundary classes Uint8. `uv` has shape `(time, element, 2)` in projected motion coordinates; `wet` has shape `(time, element)`. Optional `sourceUv` preserves original earth-relative components. The HTML frees its base64 elements after decoding. It caches reconstructed vectors only for the active pair of times.

The offline HTML requires no companion file, server, or internet. CSP disables network connections and external resources. WebGL2 shading uses native triangles and a separate instanced trail layer; a Canvas-only fallback retains finite-age particle motion and outlines. No geographic or font resources are fetched. Additive `renderer_revision: snapshot_gif_v004` and display-default fields document independent continuous motion, the continuous trail default, and the 64 MiB numeric history limit; scientific arrays and schema remain compatible.

`window.fvcomTracer` exposes runtime diagnostics and browser-test hooks (`metrics`, `select`, `setMode`, `setPlaying`, `setView`, `snapshot`) alongside the sampler and decoded arrays. `setVisualSpeed(value)` accepts 0.25–4; `setTrailDuration(seconds)` accepts 0.2–8; `setDuration(seconds)` accepts 15–300. Setters update the same state as the sliders and reject nonfinite/out-of-range values. `testAdvance(seconds)` runs the actual clock and rendering, including stall handling; `setEmissionEnabled(false)` suppresses new trail records for extinction tests while motion and time continue. `trails.alphaPixels()` reads particle-layer opacity for QA.

`metrics` reports mode/time, resets, loops, particle count, measured frame times, graphics backend, available browser memory readings, and boundary/dry/numerical stopping counters. Additions are `visualSpeed`, `trailDuration`, `playbackDuration`, `activeTime`, `stallCount`, `trailBackend`, `trailFallbackReason`, `retainedSegments`, `historyBytes`, `historyLimitBytes`, `droppedSegments`, `historyLimited`, and `effectiveHistorySeconds`. GPU history storage is additional to the reported CPU record allocation. These hooks are not presented as user controls.

Acceptance records should include exact embedded-array verification, native-value parity, reference-sampler comparisons, shader-color checks, synthetic integration tests, offline interaction checks, screenshots, and a full continuous cycle. Performance targets are measured on a named browser/backend; they are not guarantees for every device.

Snapshot export adds `gif_export` metadata (defaults, limits, and pinned encoder provenance) without changing scientific arrays. The CSP permits embedded Blob workers and Blob image previews; network connections remain disabled. `fvcomTracer.exportGif(options)` and `metrics().gifExport` are described in [gif_export.md](gif_export.md).
