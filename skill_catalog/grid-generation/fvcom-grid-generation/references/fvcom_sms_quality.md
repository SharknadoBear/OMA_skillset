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

Adaptive size fields use the production `segment_lower_envelope_hard_soft_priority` method: segment-interpolated boundary targets provide the raw control and the eight-neighbor lower envelope supplies the continuous nearshore-to-offshore transition. OBC/boundary target propagation remains the offshore control. With the default `coastal` bathymetric-gradient policy, slope-based refinement is active only within the configured distance of land/island nodes; `global` must be an explicit choice for adaptive grids.

## Conditioning Transaction Gates

- `spring-relax-v1` keeps connectivity fixed and moves only nonboundary nodes in defect-selected patches and graph halos.
- `thin-repair-v1` may flip an unprotected interior edge or split a long unprotected interior edge, then invoke regional spring relaxation.
- `aggressive-local-v2` runs target-redundant degree-3/4 pruning, superthin-pair midpoint collapse or kind-aware boundary repair, hard valence repair, and two-ring micro-relaxation before area-transition conditioning. See `local_topology_conditioning.md`.
- `area-transition-relax-v1` re-samples the Eulerian target field and processes excessive adjacent-area pairs sequentially after thin repair. Use raw area change `|A1-A2|/max(A1,A2) > 0.50`, or require target gradient, raw change, and normalized `log(A/A*)` mismatch together for a preemptive trigger.
- Reject and restore a stage if any signed area becomes nonpositive, a protected/OBC edge is lost, a new nonmanifold edge/component appears, an unedited original boundary coordinate or hard anchor moves, or controlled global quality tails regress.
- Compare every area-transition patch to the whole-stage baseline for `L/h` maximum, p95, and count above 1.55, permitting at most the explicit 0.1% numerical tolerance on maximum/p95; cap total movement at `0.25h` and retain target-normalized area-jump diagnostics.
- Record unresolved boundary-imposed defects and retain the mesh as `needs_review`; never alter the boundary to manufacture acceptance.
- Treat any remaining unique-neighbor valence above eight as a hard FVCOM readiness failure, even when the mesh remains serializable and all other structural invariants pass.
