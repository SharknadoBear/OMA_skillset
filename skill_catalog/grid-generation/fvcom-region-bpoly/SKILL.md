---
name: fvcom-region-bpoly
description: Create and geometrically adjust map-guided, feature-first four-sided RegionBPoly mission envelopes for FVCOM preprocessing, with a strict hash-bound visual gate that detects and repairs coastal land-side waterway truncation before delivery.
---

# FVCOM RegionBPoly

Use this skill to turn a regional ocean, estuary, lake, island, strait, or archipelago request into a QA-ready four-corner `RegionBPoly`.

Do not generate coastline boundary arcs, bathymetry, or meshes in this skill.

## Ownership

RegionBPoly owns mission scope, target-feature coverage, offshore-side orientation, and coastal land-side waterway completeness.

For coastal domains, this skill alone must inspect and repair waterway truncation on the three non-offshore sides. Downstream skills must not request, apply, or infer RegionBPoly adjustment.

The delivered offshore OBC is not contained by RegionBPoly. It may deform beyond the selected offshore side after passing its own topology, landfall, land-intersection, and bathymetry-support gates. The final downstream `model_domain_polygon` is authoritative for the wet domain.

`envelope_bbox` is a RegionBPoly-stage coastline-source and plotting helper. It is not a bathymetry extent.

## Primary Workflow

Run:

```powershell
python scripts/run_region_bpoly.py --request-text "..." --run-dir runs/case --name case --mode execute
```

For a fresh autonomous forward test, create a new UTC-stamped case under
`Workspace/Preprocessing/fvcom-region-bpoly/runs/`; never reuse an earlier
case directory. Use `--mode test --heuristic-mode memory` so the complete map
and review evidence remains available for an independent audit. For a named
Delaware land-side visual-gate campaign, use the directory name
`delaware_land_side_visual_gate_<UTC>`.

Modes:

- `execute` retains the provisional visual evidence until a coastal decision passes, then keeps the delivered JSON, final map, final review JSON, compact review map, and offshore-side artifact.
- `test` retains all intermediate maps and evidence.
- `--heuristic-mode auto` resolves to `memory` in execute mode and `unknown` in test mode.
- `--basemap-provider auto` is the normal map policy. `none/off` requests a real offline coastline, not a blank positive review.
- A coastal run always forces full land-side review even if fast review was requested.

The primary runner never autonomously marks a coastal candidate `pass`. It writes `final_status: needs_review` and a `region_bpoly_land_side_visual_review_request_v1` binding the exact serialized RegionBPoly, candidate JSON, whole-domain map, and every required side map by SHA-256.

Lake and island branches do not use the coastal land-side gate. Unknown requests without authorized feature geometry remain `needs_review` with `unknown_region_no_feature_plan`.

## Strict Coastal Visual Gate

After the candidate maps exist, inspect:

- the whole-domain final map; and
- start, middle, and end maps for each of the three non-offshore sides.

For every required land side, record exactly one status and concise geographic evidence:

- `pass`: connected waterways and mission features continue naturally inside the frame;
- `expand_required`: the side visibly cuts a river, estuary arm, tidal creek, strait, connected embayment, or other required wet continuation;
- `unresolved`: the evidence cannot support an autonomous pass or deterministic one-side repair.

The selected offshore side is excluded. It must never request `expand_side`.

Finalize a first-pass candidate with:

```powershell
python scripts/review_region_bpoly.py --candidate-json runs/case/region_bpoly.json --decision pass --map-visibility-status pass --side-status 0:pass --side-note "0:..." --side-status 1:pass --side-note "1:..." --side-status 2:pass --side-note "2:..." --mission-scope-status pass --single-open-boundary-status pass
```

Use the actual required side indices from the review request; the example indices are illustrative.

The finalizer verifies current hashes, readable map files, usable geographic backgrounds, exact side coverage, required features, mission scope when applicable, and offshore-side selection. Missing, stale, unusable, or incomplete evidence remains `needs_review`.

A coastal candidate is deliverable only when all required land sides pass.

## Repair Loop

At most three visual review attempts are permitted.

If exactly one land side is `expand_required`, record a `revise` decision. The review JSON emits the authorized side and next iteration. Create an explicit positive distance:

```json
{
  "operation": "expand_side",
  "side_index": 1,
  "distance_km": 10.0
}
```

Apply it:

```powershell
python scripts/adjust_region_bpoly.py --input-json runs/case/region_bpoly.json --adjustment-manifest runs/case/expand.json --output-json runs/case/region_bpoly_adjusted.json --map-path runs/case/region_bpoly_adjustment_map.png --truncation-loop
```

Then rerun the primary workflow with the same request:

```powershell
python scripts/run_region_bpoly.py --request-text "..." --input-region-json runs/case/region_bpoly_adjusted.json --land-side-review-iteration 2 --run-dir runs/case_i02 --name case_i02 --mode execute
```

`expand_side` moves only the two vertices of the named complete land side along its projected outward normal. It rejects the offshore side, nonpositive distances, invalid four-corner geometry, and self-intersection.

Inside this loop, rotation, global scaling, and free vertex reshape are forbidden. Those operations remain available only for explicit editing outside the truncation loop.

After any expansion, render fresh whole-domain and start/middle/end maps and return to the same visual gate. A nonpass third attempt, a fourth attempt, or an unusable map returns terminal `needs_review`.

## Feature and Map Policy

Every run derives target feature boxes before fitting the polygon. Required features may include upstream rivers, estuary/channel connectivity, forcing aprons, lake connections, island chains, and explicitly requested geopolitical or mission context.

Unknown or memory-disabled requests must never fall back to Delaware or another known box. Explicit `target_region_features`, required ingredients, or polygon seeds remain valid in memory-off test mode.

Use road-detail maps for small estuaries and topographic context for regional, lake, island-chain, and archipelago cases. Antimeridian domains require a compact longitude display frame.

After the strict coastal land-side gate passes, resolved tightness, obstruction, landing, and similar QA findings remain explicit nonblocking diagnostics unless they violate required-feature coverage or valid four-corner geometry.

## Domain Types

- `coastal`: one selected Atlantic/ocean-facing side for a later coastline-anchored OBC;
- `island`: offshore loop without mainland landfall anchors;
- `lake`: no ocean OBC;
- `unresolved_autonomous_failure`: no accepted geometry.

The offshore reference point selects a side only. It is not an OBC endpoint or arc.

## Workflow States for X-Ray Casting

Preserve this operational ordering and loop:

- W1 interpret request
- W2 resolve heuristic mode
- W3 derive target features
- G1 authorized feature geometry exists
- W4 seed four-corner geometry
- W5 score/refit required coverage
- G2 required features and valid geometry
- W6 select domain type and offshore side
- W7 render initial/candidate evidence
- W8 write provisional candidate and hash bindings
- W9 inspect whole-domain and land-side start/middle/end maps
- G3 land-side truncation gate
  - expand: W11 apply one authorized `expand_side`, then return to W9 with fresh maps
  - pass: W10 retain resolved QA, then W12 package final evidence, then W13 terminal delivery
  - unresolved or attempt limit: terminal `needs_review`

The five primary tool nodes are:

- T1 `run_region_bpoly.py`
- T2 feature inference/scoring modules
- T3 map rendering modules
- T4 `review_region_bpoly.py`
- T5 `adjust_region_bpoly.py`

## Outputs

A passing delivery contains:

- `region_bpoly.json`
- `region_bpoly_final_map.png`
- `offshore_boundary_artifacts.json`
- `region_bpoly_land_side_review.json` and `.png` for coastal domains

The final coastal JSON retains the review decision, iteration, side evidence, source hashes, and compact-map path. `final_status: pass` is impossible without this evidence.

## Validation

```powershell
python scripts/selftest_region_bpoly.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
