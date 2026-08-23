# FVCOM SMS 2DM Quality Reference

Use this note when changing `.2dm`, OBC nodestring, depth, constraint, or acceptance behavior.

The sole normative policy is
[`fvcom_grid_quality_policy_v1.json`](fvcom_grid_quality_policy_v1.json). Every
quality document binds its SHA-256; this reference describes how its metrics are
computed and represented.

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

## Benchmark-first decision buckets

- Class 1 blocks a first benchmark: invalid/nonfinite geometry, a structural topology break, missing protected constraints, invalid OBC/open exterior, incomplete positive-down bathymetry, node-cap or roundtrip failure, any true vertex-neighbor valence above `8`, or any superthin triangle with `q < 0.10` or minimum angle below `5 deg`.
- Class 2 is regional-refinement debt: ordinary `30–130 deg` angle tails, `q_L3sigma <= 0.75`, non-superthin quality tails, bathymetric slope above `0.1`, adjacent-area change above `0.5`, size/continuity debt, and singly connected triangles that do not create a structural break.
- Class 3 contains descriptive distributions, source/domain statistics, runtime, mesh size, and budget headroom.

Class-1 failure retains all artifacts and sets `final_status: needs_review`.
Class-2/3 findings remain visible but cannot veto topology repair or a first
benchmark run. Normal generation writes one policy-bound quality document for
the generation-time smoothed mesh.

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
- In the default minimal/autonomous path, reject and restore only if a signed area becomes nonpositive, a protected/OBC edge is lost, a new nonmanifold edge/component appears, an unedited original boundary coordinate or hard anchor moves, or the ordered valence/superthin tuple regresses. Legacy profiles may retain their documented quality-tail experiments.
- Compare every area-transition patch to the whole-stage baseline for `L/h` maximum, p95, and count above 1.55, permitting at most the explicit 0.1% numerical tolerance on maximum/p95; cap total movement at `0.25h` and retain target-normalized area-jump diagnostics.
- Record unresolved boundary-imposed defects under their policy bucket and never alter the boundary to manufacture acceptance.
- Treat any remaining unique-neighbor valence above eight as a hard FVCOM readiness failure, even when the mesh remains serializable and all other structural invariants pass.
