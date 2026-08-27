---
name: fvcom-bdry-arc
description: Create QA-ready FVCOM open-boundary arcs, continuous model loops, solid-by-default residual water closures, v2 open-exterior contracts, and mandatory Adaptive v2 boundary-resolution packages by invoking fvcom-region-bpoly for direct requests or consuming its same-run return when called as a subworkflow.
---

# FVCOM Boundary Arc

Use this skill to turn a regional FVCOM modeling request into a scientifically
auditable boundary package for `fvcom-grid-generation`.

## RegionBPoly Entry Contract

For a direct regional request, invoke `$fvcom-region-bpoly` as a returning
subworkflow. Pass the original scientific request, domain type,
feature-retention needs, OBC intent, explicit geometry, execution mode, and
heuristic context. A supplied candidate or prior RegionBPoly from an earlier
run is an input to that call, not a reason to bypass it.

When `$fvcom-grid-generation` invokes this skill as S2 and supplies the exact
canonical RegionBPoly package just returned by S1 in the same parent run,
consume that package directly and do not invoke RegionBPoly again. This is a
returned-subworkflow handoff, not generic prior-artifact reuse. Preserve the S1
paths and hashes in downstream lineage.

Use the returned `region_bpoly.json` and
`offshore_boundary_artifacts.json` verbatim. Never resize, rotate, reshape, or
regenerate the returned polygon inside this skill. Usable polygon geometry plus
a resolvable offshore orientation is sufficient to begin arc generation.
Upstream `needs_review`, package state, delivery labels, land-side findings, and
warning taxonomies remain lineage and do not independently block this workflow.
Stop only when usable geometry is absent/invalid or offshore orientation cannot
be resolved.

## Sole Boundary-Resolution Contract

`adaptive-coastal-v2` is the only active generation implementation and is
always run. Do not select a profile in normal requests or commands.
`--boundary-resolution-profile` remains only as a deprecated compatibility
argument accepting `adaptive-coastal-v2`; reject `legacy` and
`adaptive-coastal-v1`. Archived packages may be inspected downstream, but this
skill never generates those profiles.

Core boundary arcs and continuous model loops are required v2 foundations, not
alternative profiles. Read `references/oceanmesh2d_rpw2019_notes.md` before
changing topology, arc repair, island filtering, or passage rules.

## Scientific and Topological Rules

- Treat the RegionBPoly offshore point as a side selector, not a final endpoint or containment cage. A topology-valid OBC may deform outside the polygon; retain inside/outside fractions for audit.
- Use GSHHS/GSHHG polygons and physical coastline lines as the normal topology source. Full resolution `f`, level 1, is the default. Keep `cusp-legacy` only as an unrelated explicit debug/source-compatibility name.
- Require the GSHHS connector to validate selected source polygons before clipping and to record any in-memory `make_valid` repair without modifying the source cache. Reject the coastline package if selected output geometry remains invalid.
- Fetch a centered 3x RegionBPoly source footprint, never below 2x. Source-frame edges are not coastline, landfalls, or physical residual-role candidates.
- Treat projected-centroid centering and hash-bound source-manifest centering as distinct evidence: a locally projected footprint may appear shifted by projection distortion, but it is centered only when either the projected test passes or the bound topology manifest records exact zero source-center offset. Coverage factors and RegionBPoly containment remain independent hard gates.
- Require simple/nonbranching OBCs, physical landfall rules, no nonendpoint land crossing, one valid connected wet component, a continuous exterior ring, current source hashes, and the requested OBC count.
- For coastal domains, use `offshore-first` unless the request explicitly calls for `mouth-first`.
- Carry each exact `delivered_open_boundary_arc` into v2. Proximity classification is map/QA evidence only and may not restore discarded source tails.
- Preserve protected islands, mission water bodies, river context, and narrow passages. Never close a channel automatically.
- If a coastal OBC crosses blocking land away from its physical endpoints, detour around the blocking polygon in projected coordinates. Select only a simple branch that keeps the blocker inside the seeded frame, preserves the seed component, clears that blocker, and retains the full open chain; record the before/after land intersection and routing lineage.
- Scale the physical landfall/source-coastline comparison tolerance with the requested boundary target as `max(25 m, min(250 m, 0.5 h))`; this gate accommodates source/projection discretization but does not permit nonendpoint land crossing.
- After the hash-bound open-exterior contract passes, Adaptive v2 may project an endpoint-only delivered-OBC offset onto the canonical model exterior when it is no larger than one recorded repair-sampling interval. Record both endpoint distances and the limit, leave every interior coordinate unchanged, and reject larger offsets or any remaining interior exterior-overlap defect.
- Keep the aggregate edge/target limit at 1.55, enforce gradation and topology-area gates, and keep protected-passage underresolution as a hard review condition.

## Domain-Aware Adaptive v2

Coastal OBCs:

1. Keep each requested/delivered OBC as a separate ordered `LineString` with a stable integer `obc_id`.
2. Repair and sample every OBC independently.
3. Require exactly two physical hard landfall anchors per `obc_id`.
4. Equidistribute `integral(ds/h)` between landfalls and stable sharp-turn/spit-tip anchors, avoiding isolated short remainders.
5. Bisect any sampled chord whose non-landfall interior shortcuts through physical land or crosses a nonadjacent sampled chord, inserting each new point on the exact accepted source OBC until the delivered sampled chain is land-free and simple.
6. Share the same target at each land/OBC junction and grade into the complementary landward chain.

Multiple OBCs:

- Never concatenate distinct OBCs.
- Keep one continuous exterior polygon ring while tagging resolved-open features and boundary nodes with `obc_id`.
- Record per-OBC closure state, node sequence, source and sampled length, anchor counts, exterior overlap, land intersection, edge/target ratios, and gradation.

Island and archipelago OBCs:

1. Require one land-free closed exterior loop and zero landfall anchors.
2. Project native longitude/latitude coordinates directly into a compact
   antimeridian-safe metric CRS. Never translate, wrap, or otherwise rewrite
   longitudes to make the geometry appear contiguous. Densify sparse
   geographic edges before projection so the projected path follows the
   intended short circular longitude interval.
3. Place the deterministic seam at minimum projected x, using projected y and then source order as tie-breakers.
4. Add one half-perimeter balance anchor.
5. Apply the land-and-nonadjacent-chord safety guard with no endpoint exception, because a closed offshore loop has no landfalls; use a spatial index for the crossing inventory.
6. Require exactly one seam anchor, one balance anchor, complete exterior overlap, and no land intersection.

Island topology and passage safeguards:

- Compute area, perimeter, equivalent diameter, compactness, complexity, aspect, solidity, wet gap, and scale stability.
- Classify land roles with the same numerical clearance used by the loop: a required fragment whose full-clearance neighborhood connects to already external land is not an independently retainable island and inherits that external role. Propagate this relation to a fixed point and record the gaps, rounds, and affected components.
- When independently protected components are outside the seeded loop, construct their shortest projected wet-support corridors together, subtract the full land-clearance union once, and accept the result only when every protected component is enclosed, the seeded exterior stays valid, and land intersection remains zero. Use sequential fixed-point reconstruction only as a validated fallback.
- Protect mission-region islands and gaps plus the configured buffer. Preserve protected geometry exactly and require spacing no larger than one quarter of a protected gap.
- Outside protected regions, merge/drop only subgrid candidates within the 0.5% cumulative absolute island-area budget; guard validity, centroid, Hausdorff distance, area, orientation, and mission gaps.
- Inventory conservative paired-bank wet passages. Harmonize both banks from passage width and required elements across. An explicit spacing floor that underresolves a protected passage is a hard `needs_review`; unresolved unprotected passages are advisories.

## Primary Run

After the direct subworkflow or same-parent S1 handoff returns, run without a
profile option:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --coastline-gpkg gshhs_land.gpkg --coastline-source gshhs --run-dir runs/case --name case --mode test --expected-obc-count 1
```

Use `--expected-obc-count 2` for a two-opening domain and `1` for a closed
island/archipelago loop. A lake/no-ocean-boundary contract uses zero. The value
is recorded in configuration, model-loop, open-exterior, and resolution
manifests without modifying RegionBPoly.

## Residual-Water Finalization

The default `solid-default` policy writes a v2 open-exterior contract, a whole
domain map, one map per residual component, and a hash-bound pending decision.
A simple shoreline-bracketed solid closure is eligible only when it creates no
artificial bar, crosses no unrelated land, preserves the wet component, and
conflicts with no protected feature.

Before assigning any `secondary_tidal_obc`, invoke `$noaa-coops-tides` and
require a fresh, hydraulically connected station screen for that component.
Station proximity is eligibility evidence only; never exceed the requested OBC
count.

```powershell
python C:\Users\huan111\.codex\skills\noaa-coops-tides\scripts\screen_tidal_stations.py --open-exterior-contract runs/case/open_exterior/open_exterior_contract.json --wet-domain-gpkg runs/case/bdry_arc_package.gpkg --output-dir runs/case/open_exterior/coops_screen --radius-km 25
python scripts/finalize_open_exterior_decision.py --bdry-arc-manifest runs/case/bdry_arc_manifest.json --station-screen-json runs/case/open_exterior/coops_screen/noaa_coops_tidal_station_screen_v1.json --decision pass --rationale "Inspected whole-domain and component maps; accepted roles preserve physical exchange and create no artificial bar." --resume-adaptive
```

Keep `bdry_arc_package.gpkg` immutable as candidate evidence. A passing
hash-bound decision creates `bdry_arc_package_final.gpkg`, promotes accepted
secondary OBCs after the primary in residual-segment order, moves accepted
solid closures into the landward boundary with provenance, and removes those
components from the frame layer. Absorb any already-classified intentional-open
fragment into the nearest existing OBC endpoint only within the recorded hard
distance limit; keep its `obc_id`, require a simple joined line, and record the
absorbed residual segment IDs. The final package becomes canonical while the
candidate path and hash remain recorded. Rebuild model loops, final
open-exterior QA, and Adaptive v2 resolution from the finalized package. When
a later resolution attempt reaches a terminal state, replace any stale derived
parent-manifest resolution flag with that latest state rather than accumulating
duplicate review markers.

## Outputs and Assessment

Normal runs write the candidate arc package, source coverage evidence, model
loops, open-exterior contract/maps, and mandatory v2 products:

- `boundary_resolution/boundary_resolution_manifest.json`
- `boundary_resolution/boundary_resolution.gpkg`
- `boundary_resolution/boundary_resolution_diagnostics.json`
- `boundary_resolution/boundary_resolution_nodes.geojson`
- `boundary_resolution/boundary_resolution_review_map.png`

The resolution GeoPackage includes separate resolved-open rows by `obc_id`,
boundary nodes with OBC/land IDs and anchors, resolved domain/islands, and
passage diagnostics when present. Keep all evidence and return `needs_review`
when any hard scientific gate fails; do not tune geometry silently.

Long Adaptive v2 runs also write
`boundary_resolution/boundary_resolution_progress.jsonl` and
`boundary_resolution/boundary_resolution_progress_state.json`. Use the state
file for live audit and the append-only JSONL for provenance. They report the
current scientific phase, completed/total island or component counts, phase
percentage, monotonic overall percentage, elapsed time, and heartbeat sequence.
An interrupted process appends one terminal `cancelled` event with the last
completed counts instead of leaving the live state labeled `running`; an
unhandled exception instead records `failed` with its type and message.
Progress never substitutes for the final scientific manifest.

## Standalone Tools

- `scripts/analyze_boundary_resolution.py`: non-mutating v2 diagnostics.
- `scripts/refine_boundary_resolution.py`: build the mandatory v2 package from existing loop, mission, and GSHHS artifacts.
- `scripts/build_model_boundary_loops.py`: rebuild the foundational loop classification for debugging.
- `scripts/finalize_open_exterior_decision.py`: bind visual/station evidence, materialize accepted roles, and rebuild downstream v2 products.

## Validation

```powershell
python scripts/selftest_bdry_arc.py
python scripts/selftest_open_exterior_contract.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
