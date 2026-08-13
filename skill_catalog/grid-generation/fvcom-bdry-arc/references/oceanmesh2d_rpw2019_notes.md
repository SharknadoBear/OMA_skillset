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
- find the open-boundary anchors where those adjacent bpoly sides intersect the GSHHS coastline/land boundary, choosing the crossing closest to each offshore-side corner;
- use the selected bpoly offshore-side endpoints as control corners, not final anchors;
- create a smooth open-boundary arc by deforming the full seaward chain from one coastline/bpoly anchor through the offshore side to the other coastline/bpoly anchor;
- combine the smooth arc with the non-seaward bpoly path between anchors to form a closed deformed frame;
- subtract GSHHS land polygons from the deformed bpoly frame;
- choose the seeded wet-domain face rather than the largest arbitrary polygon;
- classify remaining non-open frame edges separately from GSHHS land-boundary arcs;
- keep island holes if they are resolved at the target resolution;
- mark `needs_review` when topology is ambiguous.

## Island / Archipelago Branch

Do not apply the coastline-on-bpoly mainland-anchor rule to bpoly products whose `domain_type` is `island` or whose boundary policy is `offshore_loop_no_land_anchors`. For island-chain domains, the bpoly frame is the offshore-boundary intent. Generate a smooth closed offshore loop, subtract GSHHS land polygons, keep island holes/boundaries inside the accepted water domain, and classify any loop-blocking island contact as a `land_patch_boundary` under Bear's land-patch rule. This branch protects Hawaii State, Aleutian, and similar archipelago cases from false `start/end coastline anchor missing` failures.

GSHHS full resolution (`f`) is the default topology source for this branch and for the coastal/estuary branch. Lower-resolution GSHHS products should be used only when the prompt or CLI explicitly requests them. Slow vector stages should emit progress/heartbeat artifacts rather than silently downshifting.

## CUSP Compatibility

The installed `cusp-coastline` skill writes useful EPSG:4326 `LineString` and `MultiLineString` shoreline vectors, usually in a `coastline` layer. It does not create OceanMesh2D-style `outer/mainland/inner` topology. Keep CUSP as explicit legacy/debug input or future local-detail refinement after the GSHHS topology component is known.

## Delaware / Chesapeake Separation

For Delaware River / Delaware Bay cases, use GSHHS land polygons to make Delaware/Chesapeake separation a land/topology question rather than a fragmented-line connectivity question. The seed should be in the intended Delaware wet component; CUSP should not decide broad-region connectivity in this workflow.

## Adaptive Coastal Resolution Contract

Keep the legacy boundary workflow unchanged unless `adaptive-coastal-v1` is explicit. Write adaptive products separately from legacy layers.

Use dimensionless shape and mesh-relative measures rather than raw perimeter/area alone: equivalent diameter, compactness, normalized complexity, minimum-rectangle width and aspect, solidity, and nearest wet gap divided by local target spacing.

Protect mission-region islands and gaps. Preserve protected polygon geometry exactly and propagate a target no larger than one quarter of a protected gap width. Outside protected regions, merge or drop only subgrid candidates under a cumulative absolute area budget. A candidate that requires spacing below the permitted minimum must be refined, merged, dropped, or reported; never retain an unresolved constraint silently.

Repair a coastal OBC against GSHHS land with fixed anchors before assigning nodes. Grade target spacing by arclength and build one continuous chain rather than sampling each source segment independently. Split a graded interval when its chord deviates from the repaired polyline by more than ten percent of local target size. Validate anchor equality, land avoidance for repaired and sampled paths, measured exterior overlap, chain order, and local `L/h` before downstream meshing.

Carry the exact delivered OBC and landfall lineage into adaptive processing. Resolution-scaled proximity classification may label the model exterior for maps and audits, but it must never expand the adaptive OBC or restore discarded source/coastal tails. Derive the adaptive landward chain as the complementary exterior interval between the exact delivered landfalls.

## Adaptive Coastal V2 Prevention Extension

The v2 sampler is an opt-in clean-room extension; it does not change legacy or v1 behavior. Preserve both coastline/OBC landfalls as explicit hard anchors. Detect additional sharp-turn and spit-tip anchors from local and wider-scale turn evidence, suppress nearby duplicate candidates, and equidistribute `integral(ds/h)` independently between retained anchors.

Use the same target spacing on both sides of each land/OBC junction and grade toward the land-boundary target. Inventory only conservative connectors whose interiors are covered by the accepted wet domain and whose bank tangents are compatible. A passage may lower targets on both banks, but this boundary-stage tool must not close the passage. By default, derive the permitted minimum passage spacing from the minimum protected passage width divided by the protected element count; record the controlling passage and apply the resulting targets locally on both banks. Do not substitute a fixed regional spacing floor. If an explicit user floor makes a protected passage unresolved, retain the geometry and mark it as a hard review gate. Always preserve downstream node-budget and serialization gates because an exceptionally narrow protected passage can imply expensive local refinement.
