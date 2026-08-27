---
name: fvcom-bdry-arc
description: Invoke fvcom-region-bpoly, then create QA-ready FVCOM open-boundary arcs, continuous model loops, solid-by-default residual water closures, v2 open-exterior contracts, and optional adaptive coastal boundary-resolution packages without changing the returned RegionBPoly.
---

# FVCOM Boundary Arc

Use this skill to turn a regional modeling request into a QA-ready boundary
package for later `fvcom-grid-generation`.

## RegionBPoly Subworkflow

Start every boundary-arc workflow by invoking `$fvcom-region-bpoly` as a
returning subworkflow. Do not require the caller to run it first.

Pass the original regional modeling request, intended domain type and
open-boundary intent, mission-feature or explicit-geometry inputs, offshore
orientation evidence, and selected execution/test context. If the caller
supplies candidate or prior RegionBPoly artifacts, pass them into the
subworkflow as explicit inputs; do not bypass the call.

Use the returned package directly. Prefer its canonical files when present:

- `region_bpoly.json` containing usable four-corner polygon geometry;
- `offshore_boundary_artifacts.json`;
- `region_bpoly_manifest.json`; and
- coastal land-side review JSON and map evidence when available.

Never block boundary generation because of upstream `final_status`,
`package_state`, `delivery_ready`, land-side review status, warning taxonomy,
or missing optional review evidence. Preserve those fields as provenance and
continue whenever usable polygon geometry and offshore-side orientation can be
resolved. If polygon geometry itself is absent or invalid, report that concrete
input error; there is no boundary geometry to generate. Use the returned
RegionBPoly and offshore artifacts verbatim as the primary runner inputs.

## Ownership and Core Rules

- Accept the RegionBPoly returned by the subworkflow for this run regardless of its review or delivery labels; never resize, rotate, reshape, regenerate, or request adjustment of it inside boundary-arc processing.
- Treat upstream review findings as nonblocking provenance. Boundary Arc owns and applies its own physical landfall, intersection, connected-closure, topology, source-coverage, and requested-OBC-count gates.
- Treat the RegionBPoly offshore point as a side selector, not a final endpoint or containment cage.
- Permit a topology-valid offshore OBC to deform beyond RegionBPoly. Preserve inside/outside fractions for audit only.
- For coastal estuaries, prefer `--obc-placement-policy offshore-first`: seek one complete simple offshore arc with exactly two physical-coastline landfalls and complete ownership of the open-water exterior. `mouth-first` is explicit.
- Use GSHHS/GSHHG polygons as the topology base. Keep CUSP as explicit legacy/debug input.
- Fetch a centered 3x RegionBPoly GSHHS source footprint, with 2x the hard minimum. Source-frame edges are never coastline, landfalls, or residual-role candidates.
- Preserve strict blockers for physical landfalls, nonendpoint land crossing, simple/nonbranching topology, valid connected closure, coastline-source coverage, wet-component count, and requested OBC count.
- Classify real-water residual exterior components without interpreting them as RegionBPoly truncation. A simple shoreline-bracketed solid closure is the default only when it creates no artificial bar, crosses no unrelated land, preserves the wet component, and conflicts with no protected feature.
- A nearby NOAA CO-OPS tidal station is eligibility evidence only. Never open a residual automatically and never exceed the requested OBC count.
- Preserve lake and island/archipelago branches; do not apply mainland-anchor logic to them.
- Carry the exact delivered OBC as `delivered_open_boundary_arc`. Adaptive profiles must use it and its landfalls directly.
- Keep `--boundary-resolution-profile legacy` as the default.

This skill contains no RegionBPoly mutation, adjustment candidate, or boundary-regeneration loop.

Read `references/oceanmesh2d_rpw2019_notes.md` before changing topology, arc repair, or island filtering.

## Primary Workflow

After the RegionBPoly subworkflow returns usable polygon geometry, run:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --coastline-gpkg gshhs_land.gpkg --coastline-source gshhs --run-dir runs/case --name case --mode test
```

Adaptive profiles remain opt-in:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --coastline-gpkg gshhs_land.gpkg --coastline-source gshhs --run-dir runs/case --name case --mode test --boundary-resolution-profile adaptive-coastal-v2
```

The boundary geometry is generated once from the returned RegionBPoly.

The default `--residual-boundary-policy solid-default` writes `fvcom_open_exterior_contract_v2`. `strict-reject` writes v1. v3 is historical and unsupported by active generation or validation.

The open-exterior component is non-mutating. It writes:

- `open_exterior/open_exterior_contract.json`
- `open_exterior/open_exterior_review_map.png`
- one map per residual component
- `open_exterior/open_exterior_agent_decision.json`

Every GSHHS run also writes `fvcom_coastline_source_coverage_v1`. Require central containment, at least 2x coverage on both axes, physical-coastline landfalls, zero delivered-boundary dependence on the source frame, current hashes, and whole/zoom maps.

Before finalizing residual roles, use `$noaa-coops-tides` within 25 km whenever a secondary tidal OBC is considered. Inspect the whole-domain and component maps. A passing decision assigns eligible residuals to `solid_lagoon_closure` unless an explicit, station-qualified `secondary_tidal_obc` is permitted by the requested count.

```powershell
python C:\Users\huan111\.codex\skills\noaa-coops-tides\scripts\screen_tidal_stations.py --open-exterior-contract runs/case/open_exterior/open_exterior_contract.json --wet-domain-gpkg runs/case/bdry_arc_package.gpkg --output-dir runs/case/open_exterior/coops_screen --radius-km 25
python scripts/finalize_open_exterior_decision.py --bdry-arc-manifest runs/case/bdry_arc_manifest.json --station-screen-json runs/case/open_exterior/coops_screen/noaa_coops_tidal_station_screen_v1.json --decision pass --rationale "Inspected the whole-domain and component maps; residual closures are simple, protected-feature-safe, and do not create artificial bars." --resume-adaptive
```

The finalizer verifies current map hashes, deterministic closure geometry, station-screen freshness, requested OBC count, and zero unassigned residual water. It propagates accepted solid roles as fixed landward chains.

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
- `scripts/finalize_open_exterior_decision.py`: record the mandatory hash-bound Codex map judgment, assign residual roles, bind optional CO-OPS evidence, preserve raw diagnostics, and resume adaptive construction only after the unassigned-water gate closes.

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
python scripts/selftest_open_exterior_contract.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
