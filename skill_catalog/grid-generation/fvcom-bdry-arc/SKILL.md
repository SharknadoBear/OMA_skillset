---
name: fvcom-bdry-arc
description: Create QA-ready FVCOM open-boundary arcs, continuous model loops, GSHHS-derived RegionBPoly clipping feedback, and optional adaptive coastal boundary-resolution packages. Use when Codex needs coastline-anchor arc construction, post-arc bbox correction, mainland/island topology, graded OBC nodes, or gridding-ready boundary chains before fvcom-grid-generation.
---

# fvcom-bdry-arc

Use this skill after `fvcom-region-bpoly` and before `fvcom-grid-generation`.

## Core Rules

- Treat the bpoly offshore point as a side selector, not a final endpoint.
- Build provisional coastal OBC anchors at coastline intersections on the two bpoly sides adjacent to the selected offshore side. If GSHHS wet-domain extraction shows that the source arc continues beyond the delivered exterior, trim only those source tails and promote the two delivered land/exterior intersections to the final OBC anchors.
- Use GSHHS/GSHHG polygons as the topology base. Keep CUSP as an explicit legacy/debug input.
- After model-loop construction, reject unintended GSHHS frame clipping by default. Keep GSHHS analysis in this skill and return geometry-only feedback to `fvcom-region-bpoly`; do not invent regional features here.
- Carry the exact delivered OBC into the model-loop package as `delivered_open_boundary_arc`. Adaptive profiles must use that line and its landfall anchors directly; never reconstruct the adaptive OBC from resolution-scaled proximity classifications. Retain `source_open_boundary_arc` only as a compatibility alias.
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

Bounded RegionBPoly feedback loop:

```powershell
python scripts/run_bpoly_arc_feedback_loop.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --run-dir runs/case_feedback --name case --gshhs-resolution h --gshhs-levels 1 --gshhs-lookahead-km 100 --boundary-resolution-profile adaptive-coastal-v2
```

The default `--frame-clip-policy reject-unintended` requires residual frame length no larger than `max(250 m, 0.05 * target resolution)`, residual fraction no larger than 0.001, and intended land/open exterior coverage of at least 0.999. Use `--frame-clip-policy report-only` only to reproduce historical diagnostics. When clipping fails, write `region_bpoly_arc_feedback_v1.json` and skip adaptive repair with `blocked_by_region_bpoly_feedback`.

The loop retains immutable iterations, tests at most three geometry-only candidates per adjustment, accepts at most four monotonic adjustments, and caps cumulative outward movement at 100 km per implicated side. It never changes the feature plan. Stop as `input_needs_review` when geometry adjustment cannot satisfy both RegionBPoly QA and the complete boundary gate.

`adaptive-coastal-v1`:

1. Read the exact delivered OBC and its anchors from `delivered_open_boundary_arc`, then derive only the complementary landward chain from the accepted exterior loop.
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
5. An adaptive passage-spacing floor derived from the narrowest protected passage width divided by its required elements across; do not impose a fixed default floor. Record the controlling passage, width, element count, derived spacing, and any explicit user override.
6. A `needs_review` gate when an explicit spacing floor prevents a protected passage from fitting the required elements across; unresolved unprotected passages are retained and reported as advisories.
7. Explicit anchor, junction, source-tail, passage, and adaptive-spacing metadata in diagnostics and boundary-node products.

V2 never closes a channel automatically. Its default minimum passage spacing follows the smallest protected passage rather than a regional constant, while ordinary land spacing remains independently configured. The derived fine spacing is a mesh-intent requirement, not permission to exceed downstream node or storage budgets. An unresolved unprotected passage remains unchanged and advisory-only; protected underresolution under an explicit override remains a hard gate, and geographic topology changes require a separate, evidence-backed workflow.

## Standalone Tools

- `scripts/analyze_boundary_resolution.py`: analyze an existing loop package without changing it.
- `scripts/refine_boundary_resolution.py`: write an adaptive resolution package from existing loop, mission, and GSHHS artifacts.
- `scripts/build_model_boundary_loops.py`: rebuild legacy loop classification for debugging.
- `scripts/run_bpoly_arc_feedback_loop.py`: iterate RegionBPoly, GSHHS, model loops, and adaptive-v2 under bounded geometry-only adjustment.

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
