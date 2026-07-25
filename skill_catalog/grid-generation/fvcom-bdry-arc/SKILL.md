---
name: fvcom-bdry-arc
description: Create QA-ready FVCOM open-boundary arcs, continuous model loops, and optional adaptive coastal boundary-resolution packages from RegionBPoly and GSHHG/GSHHS topology. Use when Codex needs coastline-anchor arc construction, mainland/island topology, island shape and gap diagnostics, mission-protected island generalization, graded OBC nodes, or gridding-ready boundary chains before fvcom-grid-generation.
---

# fvcom-bdry-arc

Use this skill after `fvcom-region-bpoly` and before `fvcom-grid-generation`.

## Core Rules

- Treat the bpoly offshore point as a side selector, not a final endpoint.
- Build provisional coastal OBC anchors at coastline intersections on the two bpoly sides adjacent to the selected offshore side. If GSHHS wet-domain extraction shows that the source arc continues beyond the delivered exterior, trim only those source tails and promote the two delivered land/exterior intersections to the final OBC anchors.
- Use GSHHS/GSHHG polygons as the topology base. Keep CUSP as an explicit legacy/debug input.
- Preserve lake and island/archipelago branches; do not apply mainland anchor logic to them.
- Keep `--boundary-resolution-profile legacy` as the default and preserve all legacy outputs.
- Treat the GPL OceanMesh2D snapshot as a method reference only; do not translate its MATLAB code.

Read `references/oceanmesh2d_rpw2019_notes.md` before changing topology, arc repair, or island filtering.

## Primary Workflow

Legacy behavior:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --coastline-gpkg gshhs_land.gpkg --coastline-source gshhs --run-dir runs/case --name case --mode test
```

Opt-in adaptive coastal resolution:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --coastline-gpkg gshhs_land.gpkg --coastline-source gshhs --run-dir runs/case --name case --mode test --boundary-resolution-profile adaptive-coastal-v1
```

Feature-anchored, passage-aware prevention profile:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --coastline-gpkg gshhs_land.gpkg --coastline-source gshhs --run-dir runs/case --name case --mode test --boundary-resolution-profile adaptive-coastal-v2
```

`adaptive-coastal-v1`:

1. Derive the continuous OBC portion of the accepted exterior loop.
2. Repair unintended land contact with fixed endpoints and a 250 m deterministic water-side route.
3. Grade OBC spacing from 500 m at anchors to 8 km offshore with gradation 0.15.
4. Compute island area, perimeter, equivalent diameter, compactness, complexity, aspect, solidity, gap, and scale-stability metrics.
5. Protect target-water-body and upstream-river feature polygons plus 10 km; retain protected island geometry exactly and impose `h <= gap/4` in a protected wet gap.
6. Merge or drop only unprotected subgrid candidates and stop at 0.5% cumulative absolute island-area change.
7. Generalize retained islands with area, centroid, Hausdorff, validity, principal-orientation, and mission-gap guards; split OBC chords when curvature error exceeds 10% of local target size.
8. Write a separate explicit-chain resolution package; never overwrite legacy loop layers.

`adaptive-coastal-v2` retains the v1 topology and island safeguards, then adds:

1. Exact hard anchors at both OBC landfalls plus stable sharp turns and spit tips.
2. Anchor-to-anchor metric equidistribution, avoiding an isolated short remainder edge.
3. One shared land/OBC junction target with the configured gradation into the land chain.
4. A conservative wet-passage inventory with paired-bank spacing harmonization.
5. A `needs_review` gate for protected passages that cannot fit four elements across at the permitted minimum spacing; unresolved unprotected passages are retained and reported as advisories.
6. Explicit anchor, junction, source-tail, and passage metadata in diagnostics and boundary-node products.

V2 never closes a channel automatically. An unresolved unprotected passage remains unchanged and advisory-only; protected underresolution remains a hard gate, and geographic topology changes require a separate, evidence-backed workflow.

## Standalone Tools

- `scripts/analyze_boundary_resolution.py`: analyze an existing loop package without changing it.
- `scripts/refine_boundary_resolution.py`: write an adaptive resolution package from existing loop, mission, and GSHHS artifacts.
- `scripts/build_model_boundary_loops.py`: rebuild legacy loop classification for debugging.

## Outputs

Every normal run retains the legacy boundary-arc and model-loop outputs. Adaptive runs additionally write:

- `boundary_resolution/boundary_resolution_manifest.json`
- `boundary_resolution/boundary_resolution.gpkg`
- `boundary_resolution/boundary_resolution_diagnostics.json`
- `boundary_resolution/boundary_resolution_nodes.geojson`
- `boundary_resolution/boundary_resolution_review_map.png`

Require final OBC anchors at the delivered land/exterior intersections, measured complete delivered-exterior overlap, no non-endpoint land intersection in the delivered or sampled geometry, a valid resolved wet domain, zero protected-region topology operations, and area change within budget. Preserve any discarded source-arc tail and its intersections as provenance advisories rather than applying delivered-boundary gates to it. For v2, additionally require exactly two OBC-landfall hard anchors, boundary edge-to-target ratio no greater than 1.55, and no unresolved protected passage. Keep artifacts and mark `needs_review` when a hard guard fails; record unresolved unprotected passages as advisories.

## Validation

```powershell
python scripts/selftest_bdry_arc.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
