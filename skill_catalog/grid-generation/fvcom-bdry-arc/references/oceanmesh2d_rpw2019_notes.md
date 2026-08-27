# OceanMesh2D / RPW2019 Notes for fvcom-bdry-arc

Use this reference when changing coastline topology, island filtering, seeded wet-domain extraction, or open-boundary arc classification.

## Local Reference

- OceanMesh2D snapshot: `Resources/OceanMesh2D_Projection_snapshot/`
- Provenance: `Resources/OceanMesh2D_Projection_snapshot_PROVENANCE.md`
- Repository: `https://github.com/CHLNDDEV/OceanMesh2D`
- Branch: `Projection`
- Commit recorded locally: `754d69a629d7b326383665123e7ea879d9db7040`
- Paper: Roberts, Pringle, and Westerink, 2019, OceanMesh2D 1.0, GMD.
- Licensing note: OceanMesh2D is GPL-3.0. Use it as a method reference unless Bear explicitly approves license-aware code reuse.

## Method Translation

OceanMesh2D's automatic boundary-definition idea is topology-building, not simple line labeling:

- build a bbox or frame closure for open shoreline data;
- split shoreline into components;
- classify components as mainland/outer boundary or inner island holes by relation to the frame;
- remove unresolved islands with resolution-scaled thresholds;
- densify and smooth boundary vertices according to target grid spacing;
- use the resulting outer plus inner geometry for inside/outside signed-distance behavior.

For `fvcom-bdry-arc`, translate that idea into a Python postprocessor:

- use GSHHS/GSHHG closed land polygons as the default topology base;
- keep derived GSHHS coastline lines as anchor and arc-contact evidence;
- project to a local CRS before distance, tangent, and area decisions;
- bridge dangling endpoints only under strict geometric rules;
- use the selected bpoly offshore side to identify the two adjacent bpoly sides;
- find open-boundary anchors where those adjacent bpoly sides intersect the physical GSHHS `coastline_lines`, choosing the crossing closest to each offshore-side corner;
- use the selected bpoly offshore-side endpoints as control corners, not final anchors;
- create a smooth open-boundary arc by deforming the full seaward chain from one coastline/bpoly anchor through the offshore side to the other coastline/bpoly anchor;
- if that full chain crosses isolated blocking land away from its endpoints, route around a projected clearance buffer and choose the simple branch that keeps the blocker inside the seeded frame, retains the seed component, and clears the crossing; never solve this by discarding the source tail or excluding the island;
- combine the smooth arc with the non-seaward bpoly path between anchors to form a closed deformed frame;
- subtract GSHHS land polygons from the deformed bpoly frame;
- choose the seeded wet-domain face rather than the largest arbitrary polygon;
- classify remaining non-open frame edges separately from GSHHS land-boundary arcs;
- keep island holes if they are resolved at the target resolution;
- mark `needs_review` when topology is ambiguous.

Never rebuild physical shoreline from the boundary of bbox-clipped land polygons.
That boundary includes artificial source-frame segments. Keep the land polygons
for mask subtraction, but use only source-derived coastline lines for landfalls,
arc trimming, and shoreline-bracketed residual roles. Require a centered source
footprint before any of those decisions.

Validate selected GSHHS source polygons before clipping. Repair only invalid
in-memory polygonal features with `make_valid`, retain polygonal components,
derive land and coastline from that validated geometry, and preserve source
component hashes as proof that the cache was not edited. Record validity
reasons, repair method, equal-area change, and post-repair validity.

A centered source footprint may have a shifted centroid after local projection.
Accept centering from projection geometry or from an exact-zero center offset
in the hash-bound GSHHS topology manifest, but never waive the independent 2x
coverage and RegionBPoly-containment gates. Compare physical landfalls to the
source coastline with `max(25 m, min(250 m, 0.5 h))`; nonendpoint OBC/land
intersection remains a zero-tolerance topology condition.

## Island / Archipelago Branch

Do not apply the coastline-on-bpoly mainland-anchor rule to bpoly products whose `domain_type` is `island` or whose boundary policy is `offshore_loop_no_land_anchors`. For island-chain domains, the bpoly frame is the offshore-boundary intent. Generate a smooth closed offshore loop, subtract GSHHS land polygons, keep island holes/boundaries inside the accepted water domain, and classify any loop-blocking island contact as a `land_patch_boundary` under Bear's land-patch rule. This branch protects Hawaii State, Aleutian, and similar archipelago cases from false `start/end coastline anchor missing` failures.

GSHHS full resolution (`f`) is the default topology source for this branch and for the coastal/estuary branch. Lower-resolution GSHHS products should be used only when the prompt or CLI explicitly requests them. Slow vector stages should emit progress/heartbeat artifacts rather than silently downshifting.

Adaptive v2 progress must expose completed and total counts for source-island
metrics, subgrid actions, island generalization, passage components, and
boundary sampling. Preserve both an append-only event log and an atomic current
state with monotonic overall percentage and elapsed time. A progress percentage
is operational evidence only; scientific acceptance still comes from the final
resolution manifest and its topology, spacing, and passage gates.

## CUSP Compatibility

The installed `cusp-coastline` skill writes useful EPSG:4326 `LineString` and `MultiLineString` shoreline vectors, usually in a `coastline` layer. It does not create OceanMesh2D-style `outer/mainland/inner` topology. Keep CUSP as explicit legacy/debug input or future local-detail refinement after the GSHHS topology component is known.

## Delaware / Chesapeake Separation

For Delaware River / Delaware Bay cases, use GSHHS land polygons to make Delaware/Chesapeake separation a land/topology question rather than a fragmented-line connectivity question. The seed should be in the intended Delaware wet component; CUSP should not decide broad-region connectivity in this workflow.

## Mandatory Adaptive v2 Topology Foundation

Adaptive v2 is the sole active boundary-resolution workflow. Preserve the core
arc/model-loop artifacts as source evidence and write the resolved products
separately; they are v2 foundations, not selectable legacy/v1 profiles.

Use dimensionless shape and mesh-relative measures rather than raw perimeter/area alone: equivalent diameter, compactness, normalized complexity, minimum-rectangle width and aspect, solidity, and nearest wet gap divided by local target spacing.

Protect mission-region islands and gaps. Preserve protected polygon geometry exactly and propagate a target no larger than one quarter of a protected gap width. Outside protected regions, merge or drop only subgrid candidates under a cumulative absolute area budget. A candidate that requires spacing below the permitted minimum must be refined, merged, dropped, or reported; never retain an unresolved constraint silently.

Repair a coastal OBC against GSHHS land with fixed anchors before assigning nodes. Grade target spacing by arclength and build one continuous chain rather than sampling each source segment independently. Split a graded interval when its chord deviates from the repaired polyline by more than ten percent of local target size. Validate anchor equality, land avoidance for repaired and sampled paths, measured exterior overlap, chain order, and local `L/h` before downstream meshing.

Carry every exact delivered OBC and its `obc_id`/landfall lineage into v2.
Resolution-scaled proximity classification may label the model exterior for
maps and audits, but it must never expand an OBC or restore discarded
source/coastal tails. Derive each intervening landward chain as the
complementary exterior interval between exact delivered chains, preserving one
continuous exterior polygon ring.

During hash-bound residual finalization, keep the candidate GeoPackage as
evidence. Materialize each accepted secondary OBC as a separate chain, move
accepted solid closures into the landward boundary, and remove both from the
frame layer. An intentional-open fragment may be joined only to the nearest
existing OBC endpoint within the hard distance limit; the join must remain a
simple line and retain its OBC ID and absorbed-segment provenance. Rebuild the
model loop, open-exterior QA, and Adaptive v2 package from the role-resolved
GeoPackage rather than reusing candidate frame-length gates.

For a closed island/archipelago OBC, do not use landfall repair. Require the
delivered loop to equal the exterior. Densify sparse geographic edges along
their shortest circular longitude interval, then project the original native
coordinates directly into the producer-recorded compact metric CRS; never
translate or warp longitude values. Rotate the projected loop to the minimum-x
seam (projected y then source order as tie-breakers), and add the half-perimeter
balance anchor. The loop must have zero landfall anchors and remain land-free.

Use the loop's numerical land clearance when assigning island/external roles.
If a nominally required component is connected to already external land within
the full clearance diameter, the loop cannot physically pass between them; it
therefore inherits the external role. Propagate that relation to a fixed point
and retain the component gaps and role lineage in the manifest. This is a
topological separability test, not an area-based island-removal heuristic.

If independently protected land components remain outside the seeded loop,
build all shortest projected wet-support corridors simultaneously and subtract
the complete land-clearance union once. Accept the batch result only when it is
valid, contains the wet seed, encloses every protected component, and has zero
land intersection. A sequential fixed-point construction may be used only as a
fallback subject to the same whole-result gates.

## Adaptive v2 Anchors and Passage Prevention

For each coastal `obc_id`, preserve both coastline/OBC landfalls as explicit
hard anchors. Detect additional sharp-turn and spit-tip anchors from local and
wider-scale turn evidence, suppress nearby duplicate candidates, and
equidistribute `integral(ds/h)` independently between retained anchors. Never
concatenate separate OBCs.

Use the same target spacing on both sides of each land/OBC junction and grade toward the land-boundary target. Inventory only conservative connectors whose interiors are covered by the accepted wet domain and whose bank tangents are compatible. A passage may lower targets on both banks, but this boundary-stage tool must not close the passage. Derive the permitted minimum passage spacing from the minimum protected passage width divided by the protected element count; record the controlling passage and apply the targets locally on both banks. Do not substitute a fixed regional spacing floor. If an explicit user floor makes a protected passage unresolved, retain the geometry and mark it as a hard review gate. Always preserve downstream node-budget and serialization gates because an exceptionally narrow protected passage can imply expensive local refinement.

For many-island domains, use expanded component envelopes only as the
conservative passage-pair broad phase. Apply exact nearest-distance and wet
connector tests to every retained candidate; the index may admit false
positives for speed but may not remove a pair whose exact separation is within
the configured maximum width.
