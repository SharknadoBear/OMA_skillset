# Scientific and visual interpretation

## Coordinates and vectors

Retain source element numbering and node IDs. Internally normalize triangle orientation to CCW and build edge adjacency indexed by the opposite vertex. Plot in local metric coordinates after subtracting a fixed origin; this preserves GPU precision with large projected coordinates.

Projected source x/y require a metre-based CRS. Recover geographic labels with pyproj; reject disagreement greater than 20 m when a valid source lon/lat pair is also present. A geographic-only regional mesh uses a local azimuthal equidistant projection. Grid-relative currents require their projected grid CRS. East/north currents use the local projection Jacobian, evaluated with one-metre geodesic perturbations. Preserve their original components separately for native hover inspection.

The Galveston April 2025 outputs have zero-filled lon/lat, native UTM 15N coordinates, and Cartesian ua/va. Their existing preconfiguration manifests establish EPSG:32615. Single-output coordinate precision introduces at most about a decimetre relative to the double-precision input geometry; measure this when comparing files.

## Spatial reconstruction

For each wet connected fan of elements incident on a vertex, compute the area-weighted mean of finite element velocity components. Do not average across disconnected fans separated by dry cells. Store the resulting values per triangle corner, then use barycentric interpolation within that original triangle. Shade the magnitude of this vector, using the same corner values and interpolation as the particle sampler.

This is a smooth visualization reconstruction of FVCOM's element-centered velocity, not a conservative replacement for the model's flux discretization. Native element components and display vectors are both inspectable. No rectangular regridding, interpolation across an island, or missing-value filling is performed. Nonfinite wet vectors are excluded and counted in provenance. An all-invalid frame fails.

## Particle integration and boundaries

Use midpoint/RK2 steps limited to 0.35 of local minimum triangle altitude divided by speed; re-evaluate at the midpoint. Trace every proposed segment across adjacent triangle edges. Terminate at an exterior edge, dry cell, or failed walk, and reseed without drawing a connecting line. Draw accepted integration subsegments rather than the potentially unsafe chord connecting the frame endpoints. A 256-substep/walk bound terminates a particle and records the event; it never permits a boundary jump.

Seeds are area-weighted in visible wet triangles with a reproducible xorshift seed. Eight-second active-playback lifetimes and respawning preserve coverage. Particle updates use 60 Hz ticks, with adaptive RK2 substeps inside each tick. These lifecycle choices are illustrative and do not conserve particle number or represent releases.

Retain timestamped, accepted integration segments rather than fading a bitmap recursively. For segment age a and maximum duration T, opacity is `(exp(-3*a/T)-exp(-3))/(1-exp(-3))` for a<T and zero thereafter. WebGL2 instanced segments evaluate age along each segment. Canvas fallback approximates the envelope using eight age buckets but strictly removes expired segments. Both clear and redraw the transparent layer each frame, so low-alpha rounding cannot retain old pixels. The numeric history ring is capped at 64 MiB (with a corresponding GPU buffer on WebGL2); overflow drops oldest records and discloses a shortened effective window. Defaults are 3 s for snapshot and 1 s for continuous. Shortening T removes old history immediately; increasing it cannot restore previously expired segments.

## Time

Snapshot mode uses exactly one source field and an explicit display acceleration; model time remains fixed. Continuous mode linearly interpolates components on the displayed model clock. Playback duration alone controls that clock; the default complete interval takes 60 seconds. An independent Visual speed multiplier M gives `dx/dt_model = M*u(x,t_model)`, with corresponding y motion. Apply M to first/midpoint RK2 velocities and the spatial step limit, never to the time argument of the sampler. At 1× motion uses model velocity; other values are illustrative displacement, visibly labeled. Colors and hover remain unscaled. Integration splits at source timestamps; particles survive ordinary changes of source pair and speed changes.

Between source records use only cells valid and wet at both endpoints, reconstructing both fields with that common mask. Clear old trails on a mask transition to avoid retaining trails over newly dry cells. Scrubbing, mode changes, and loop wrap clear and reseed. Never connect the final time to the first as if the data were periodic.

Requested start/end times are inclusive native timestamps. Across multiple stacks, identical duplicate records are deduplicated; conflicting values, masks, or geometry fail. The default missing-record threshold is 1.5 times the smallest positive selected interval; `--max-gap-hours` overrides it. Irregular outputs need an explicitly justified threshold. With too few timestamps an absent intermediate record cannot be inferred from cadence alone. Playback pauses before an identified gap, without interpolating across it. Pause and hidden tabs freeze clocks and trail age. A visible stall above 250 ms advances the field clock across covered records, respecting gaps and loops, then clears and reseeds instead of integrating a large particle backlog.

## Map and color

Mesh boundary edges have one incident element. They include island outlines, physical coastline approximations, and artificial domain cuts. An optional obc.dat marks an edge open only when both endpoints are listed open nodes. Other edges use solid styling, describing model boundary classification rather than a new shoreline survey.

The color scale is fixed over the whole selected interval. Both Galveston review products use 0–1.6 m/s and square-root normalization, including their first-day maxima while resolving weak interior flow. Pale tails provide direction; the scalar legend provides speed. Values above a user-selected maximum saturate at the final color but remain available numerically.

Continuous mode is a first-draft, three-hour-output interpolation for visual exploration. It cannot recover unresolved tidal variability or establish validated Lagrangian trajectories. No wind, diffusion, vertical particle movement, or particle mass is modeled.

## Snapshot presentation GIFs

Export uses the same native-triangle shader and shared snapshot integrator as the viewer, with independent reproducible particles. The selected scientific timestamp stays fixed. High-resolution output increases pixel density while retaining logical viewport scale, so visual acceleration does not change with output width. A 60 Hz integration clock is sampled at 10 or 20 FPS after a trail-duration warm-up. GIF palette quantization approximates colors with at most 255 opaque entries; velocities, the numeric legend, and map geometry are unscaled. These are illustrative presentation animations, not particle-tracking outputs. See [gif_export.md](gif_export.md).
