---
name: fvcom-regional-grid-refinement
description: Deterministically refine an element-aligned region of an existing FVCOM/SMS triangular mesh while preserving protected node IDs, boundary chords, exterior topology, bathymetry lineage, and FVCOM preconfiguration contracts. Use for dredging corridors, estuary subregions, river mouths, channels, or other local-resolution upgrades that must be stitched back into a legacy mesh without changing OBC or river-node identities.
---

# FVCOM Regional Grid Refinement

Create an isolated, evidence-rich regional refinement of an existing FVCOM/SMS mesh. Treat the original mesh and every protected identifier as immutable inputs. Promote only a deterministic Gmsh Frontal-Delaunay-6 result that passes seam, topology, quality, resolution, numbering, bathymetry, and round-trip checks.

## Required companion skills

Read and follow `$fvcom-grid-generation` before generating a patch mesh. Its quality policy, Gmsh-6 backend, and `minimal-topology-v1` conditioning rules are authoritative. Read and follow `$fvcom-preconfiguration` before writing FVCOM DAT files. Do not duplicate or weaken either policy in a project adapter.

## Inputs

Require a configuration containing:

- an existing SMS 2DM mesh or equivalent FVCOM grid/depth pair;
- a projected CRS suitable for metre-based distances;
- a refinement polygon;
- an optional centerline with width or cross-feature vertex-column targets;
- protected node IDs and optional ordered OBC/river contracts;
- a patch buffer, gradation, node cap, and size bounds;
- an optional `physical_boundary_refinement` request; omitted requests default to `mode: locked`;
- one or more bathymetry sources plus blending rules;
- an isolated output workspace.

Validate the request with `scripts/validate_request.py`. Use `references/request_schema.json` as the contract. Stop instead of guessing a missing protected-node or boundary-role contract.

## Operational workflow

1. **Freeze the source.** Checksum-copy the mesh, polygons, bathymetry, and protected contracts into the output workspace. Record dependencies and the active `$fvcom-grid-generation` policy hash.
2. **Extract an element-aligned patch.** Select every source element intersecting the buffered refinement polygon. Close vertex-touch ambiguities by absorbing all source elements incident to ambiguous boundary nodes until every patch-boundary node has degree two. Preserve every resulting boundary chord exactly.
3. **Classify boundaries.** Separate physical source-boundary chords from interior stitch chords. Order the outer loop and island loops. Export the core, geometric buffer, actual patch, loops, stitch edges, and protected nodes for review.
4. **Resolve optional physical-boundary refinement.** Keep `mode: locked` by default. For `selected_exact_chord_split`, match the supplied selector to source physical chords within the configured tolerance, reject OBC/stitch/river/protected incidence, and split only chords with `L/h` above the configured maximum. Use `ceil[L/(maximum_ratio*h_mid)]` exact collinear sub-chords. Preserve every source vertex, record parent edge/fraction lineage, and absorb complete incident wet-element stars before repeating loop closure if a selected chord was not already on the patch boundary. Never split an interior stitch chord.
5. **Build the size intent.** Use geometry-derived targets such as `width / (vertex_columns - 1)`. Clip only the feature-core target. Form `min(existing, core_target + gradation * distance_to_core)`, then apply a wet-domain graph lower envelope. Audit the stitch field against the incumbent field with p95 at most 1.5 and maximum at most 2.0. Expand the patch rather than relax these gates. Audit delivered physical-boundary `L/h` separately.
6. **Generate only Gmsh-6 operationally.** Use one thread, seed 1, first-order triangles, eight native smoothing steps, the resolved locked or selected-exact-chord boundary discretization, and no algorithm fallback. Continuous callbacks must evaluate the exact refinement polygon so a coarse source triangle with no interior source vertex cannot hide the target.
7. **Audit resolution.** Evaluate the requested effective vertex columns across the feature on explicit stations. If insufficient, multiply the core target by 0.9 for no more than two retries. Keep every attempt immutable.
8. **Merge deterministically.** Delete only patch-interior source nodes/elements. Reuse vacated IDs in coordinate-sorted order, then append above the source maximum. Preserve every retained original ID and coordinate. Preserve protected OBC and river IDs exactly. Treat inserted exact-chord nodes as new deterministic nodes with explicit boundary lineage.
9. **Condition minimally.** Apply only `$fvcom-grid-generation` `minimal-topology-v1` legal local topology operations. Do not move nodes, globally retriangulate, spring, or area-smooth. Delivered boundary sub-edges and protected nodes are immutable.
10. **Remap bathymetry.** Preserve exact source values at retained exterior and stitch nodes. Interpolate exact-chord node depths linearly from their parent-edge endpoint depths. Interpolate other new nodes from configured sources, apply explicit tapers, and write scenario formulas with provenance. Never silently apply a datum offset or extrapolation.
11. **Write and validate.** Produce topology-identical scenario meshes and, when requested, six-file FVCOM preconfiguration packages. Require manifold connectivity, intended boundary components, positive areas, no `q < 0.10`, no minimum angle below 5 degrees, valence at most eight, exact seams, contiguous IDs, exact protected contracts, exact boundary geometry/length, physical-boundary size gates, formula closure, and exact 2DM read/write round-trip.

## Reusable helpers

- `scripts/refinement_core.py` contains deterministic topology, numbering, seam, quality, resolution, exact-chord splitting, augmented-loop, boundary-lineage, boundary-size-audit, and bathymetry helpers for project adapters.
- `scripts/validate_request.py` validates the generic JSON request contract.
- `scripts/self_test.py` runs offline synthetic tests for interior patches, coastline-touching patches with an island, exact-chord geometry, protected-edge rejection, deterministic numbering, augmented incident stars, seam rejection, bathymetry interpolation, and repeatable hashes.

Run:

```powershell
python scripts/self_test.py
python scripts/validate_request.py --request path/to/request.json
```

The helper layer does not replace the installed Gmsh-6 generation or FVCOM preconfiguration skills. Project workflows should import it and retain their own immutable attempt directories, scientific configuration, and domain-specific bathymetry adapter.

## Promotion rules

Retain every candidate, including failures. Publish a technical success only when every blocking refinement check passes. Keep datum, forcing compatibility, and model-execution readiness as separate gates. Never distribute a refined mesh into production cases unless the user separately authorizes that integration.

## Forward-test evidence

Read both `references/delaware_forward_test.json` and `references/delaware_v3_forward_test.json`. The original evidence records the locked-boundary workflow. The v3 evidence records the optional selected-exact-chord fallback, a uniform four-column corridor target, unchanged OBC 1-139 and ten river IDs, balanced coastal-area gates, deterministic Gmsh-6 generation, minimal topology conditioning, four bathymetry scenarios, and terminal validation. These records are workflow evidence, not universal scientific configurations.

Read `references/delaware_forward_test.json` when checking expected behavior. That evidence records a coastline-touching, four-loop Delaware patch with preserved OBC 1–139 and ten river IDs, deterministic Gmsh-6 generation, minimal topology conditioning, four bathymetry scenarios, and terminal validation. It is evidence of the workflow, not a universal scientific configuration.
