# FVCOM SMS 2DM Quality Reference

Use this note when changing `.2dm`, OBC nodestring, depth, constraint, or acceptance behavior.

## Output Invariants

- Write `MESH2D`, `MESHNAME`, `E3T`, `ND`, and `NS` records.
- Keep node depths finite and positive down.
- Keep triangles counterclockwise with positive projected area.
- Write one ordered `NS` nodestring unless upstream metadata explicitly defines no ocean boundary.
- Require every consecutive OBC node pair to be an actual mesh boundary edge.
- Preserve every constraint selected by the postprocessing boundary policy.
- Keep every pre-existing land, island, frame, and open-boundary coordinate bitwise fixed during guarded-v1 conditioning. Under the explicit `aggressive-local-v2` protocol, keep all hard anchors fixed and require exact coordinates for every surviving original boundary node; only recorded kind-aware split/removal transactions may change boundary discretization.
- Verify the serialized 2DM roundtrip preserves connectivity and OBC order, keeps coordinate shifts below `0.01 m`, and retains strictly positive projected signed area for every triangle.
- Write `delivered_boundary_nodes.geojson` from the terminal constraint chains so recovery nodes are represented explicitly.
- Permit topology edits only on local cavities; never rebuild the full Delaunay triangulation during spring relaxation, thin repair, pruning, or valence repair.

## Default Gates

- minimum triangle angle: `30 deg`;
- maximum triangle angle: `130 deg`;
- maximum bathymetric slope: `0.1`;
- maximum adjacent element area-change metric: `0.5`;
- maximum true vertex-neighbor valence: `8`;
- one manifold, traversable component;
- no missing protected constraints or nonpositive elements.

When any gate fails, retain all artifacts and set `final_status: needs_review`. Normal generation writes one quality document for the generation-time smoothed mesh. Cleanup comparisons belong to the standalone postprocessor.

Adaptive boundary packages additionally require ordered explicit chains, per-node target spacing, and OBC size compatibility: 95th-percentile `L/h <= 1.55` and maximum `L/h <= 2.0`.

All boundary profiles use the single `fvcom_size_field_v4` production method.
Construct its nearshore target as

```text
h_N = gradate_wet(min(h_S, h_G, h_H))
```

where `h_S` is the segment-interpolated solid-boundary target plus its
land-distance background, `h_G` is the bathymetric-gradient target inside the
coastal mask, and `h_H` is the geometry-derived hydraulic-corridor target.
Use delivered adaptive solid-boundary targets directly; do not replace a finer
delivered value with the configured legacy land-spacing default.

Detect the hydraulic skeleton from raster Voronoi-label discontinuities between
opposing, nonlocal solid-boundary segments. Never treat the OBC as a bank.
Reject spans that exceed the configured width, fail the opposing-bank angle,
connect locally adjacent contacts on the same boundary chain, or cross land or
an island hole. Integrate depth across each accepted paired-bank chord to
estimate cross-section area. Rank the wet-distance storage-over-cross-section
proxy in log space to obtain `I` in `[0,1]`, then set

```text
N_perp = N_min + (N_max - N_min) I
h_skeleton = clip(W / N_perp, h_min, h_max)
```

This importance is a tidal-exchange ranking proxy, not a solved velocity; its
storage accumulation has branch and loop ambiguity. Limit longitudinal size
change on the skeleton, propagate its target through connected wet cells, and
blend transversely in log space from `h_S` at the bank to `h_skeleton` at the
medial axis.

For an open domain, propagate both distance `d_wet` and the originating
delivered OBC target `h_open` through connected wet cells. Define

```text
xi    = clip((d_wet - L_hold) / L_transition, 0, 1)
P(xi) = 6 xi^5 - 15 xi^4 + 10 xi^3
h_T   = exp((1-P) log(h_open) + P log(h_N))
```

Hold OBC authority through `L_hold`, use the configured transition length
without automatic extension, and report the theoretical derivative-limited
length, available wet distance, and any post-gradation hold debt. Closed domains
use `h_N` directly. Clip to physical bounds and apply the final eight-neighbour
lower gradation envelope only over the wet domain; it may refine but never
coarsen a cell. CFL remains diagnostic only. Preserve the component map,
hydraulic masks and metrics, wet-OBC distance/target arrays, source attribution,
and transition audit in `size_field.nc` and `size_field_components.png`.

## Conditioning Transaction Gates

- `spring-relax-v1` keeps connectivity fixed and moves only nonboundary nodes in defect-selected patches and graph halos.
- `thin-repair-v1` may flip an unprotected interior edge or split a long unprotected interior edge, then invoke regional spring relaxation.
- `aggressive-local-v2` runs target-redundant degree-3/4 pruning, superthin-pair midpoint collapse or kind-aware boundary repair, hard valence repair, and two-ring micro-relaxation before area-transition conditioning. See `local_topology_conditioning.md`.
- `area-transition-relax-v1` re-samples the Eulerian target field and processes excessive adjacent-area pairs sequentially after thin repair. Use raw area change `|A1-A2|/max(A1,A2) > 0.50`, or require target gradient, raw change, and normalized `log(A/A*)` mismatch together for a preemptive trigger.
- Reject and restore a stage if any signed area becomes nonpositive, a protected/OBC edge is lost, a new nonmanifold edge/component appears, an unedited original boundary coordinate or hard anchor moves, or controlled global quality tails regress.
- Compare every area-transition patch to the whole-stage baseline for `L/h` maximum, p95, and count above 1.55, permitting at most the explicit 0.1% numerical tolerance on maximum/p95; cap total movement at `0.25h` and retain target-normalized area-jump diagnostics.
- Record unresolved boundary-imposed defects and retain the mesh as `needs_review`; never alter the boundary to manufacture acceptance.
- Treat any remaining unique-neighbor valence above eight as a hard FVCOM readiness failure, even when the mesh remains serializable and all other structural invariants pass.
