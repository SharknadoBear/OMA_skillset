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
- use the bpoly offshore side to find two coastline anchors;
- create a smooth open-boundary arc between anchors;
- polygonize GSHHS coastline boundaries plus the selected arc and bpoly frame where feasible;
- subtract GSHHS land polygons from the selected water face;
- choose the seeded wet-domain face rather than the largest arbitrary polygon;
- keep island holes if they are resolved at the target resolution;
- mark `needs_review` when topology is ambiguous.

## CUSP Compatibility

The installed `cusp-coastline` skill writes useful EPSG:4326 `LineString` and `MultiLineString` shoreline vectors, usually in a `coastline` layer. It does not create OceanMesh2D-style `outer/mainland/inner` topology. Keep CUSP as explicit legacy/debug input or future local-detail refinement after the GSHHS topology component is known.

## Delaware / Chesapeake Separation

For Delaware River / Delaware Bay cases, use GSHHS land polygons to make Delaware/Chesapeake separation a land/topology question rather than a fragmented-line connectivity question. The seed should be in the intended Delaware wet component; CUSP should not decide broad-region connectivity in this workflow.
