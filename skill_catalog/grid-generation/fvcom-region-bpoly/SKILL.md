---
name: fvcom-region-bpoly
description: Create and geometrically adjust map-guided, feature-first four-sided RegionBPoly mission envelopes for FVCOM preprocessing, using hash-bound coastal review and bounded land-side repair while always returning the latest valid geometry with explicit warnings.
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

- `execute` retains provisional visual evidence through coastal refinement. A clean pass may discard intermediates; a best-effort acceptance retains them with the review warnings.
- `test` retains all intermediate maps and evidence.
- `--heuristic-mode auto` resolves to `memory` in execute mode and `unknown` in test mode. `unknown` bypasses catalog geometry but still permits named-place discovery.
- `--place-discovery auto` is the default. When no catalog or explicit feature geometry exists, make one cached OpenStreetMap Nominatim lookup for the named geographic target, retain its attribution and selected bbox in `region_place_discovery.json`, and use it only as an initial visual seed.
- If automatic lookup is unavailable or ambiguous, research or infer a reasonable initial frame and pass it with `--discovery-bbox WEST SOUTH EAST NORTH` plus `--discovery-label`. The visual maps, not the guessed precision, decide whether the seed is acceptable.
- `--basemap-provider auto` is the normal map policy. `none/off` requests a real offline coastline, not a blank positive review.
- A coastal run always forces full land-side review even if fast review was requested.

The primary runner writes the nonterminal state `final_status: review_pending` and a `region_bpoly_land_side_visual_review_request_v1` binding the exact serialized RegionBPoly, candidate JSON, whole-domain map, and every required side map by SHA-256.

Never end the skill at `review_pending`, `repair_required`, or `needs_review`. Continue through visual finalization and any authorized repairs. Once a valid four-corner geometry exists, the finalizer always returns it as `final_status: pass`: cleanly when the evidence passes, or as an accepted best-effort delivery with explicit warnings when visual evidence is unresolved, stale, unavailable, or still nonpassing after the repair limit. Only failure to obtain usable polygon geometry may stop without a delivery.

Lake and island branches do not use the coastal land-side gate. Unknown named regions must enter place discovery instead of terminating at G1.

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

The finalizer verifies current hashes, readable map files, usable geographic backgrounds, exact side coverage, required features, mission scope when applicable, and offshore-side selection. Missing, stale, unusable, or incomplete evidence is retained as a nonblocking delivery warning; it never changes a valid final RegionBPoly to `needs_review`.

A clean coastal review passes only when all required land sides pass. If they cannot all pass autonomously, the latest valid geometry is still deliverable as best effort so downstream processing can apply its own physical gates.

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

After any expansion, render fresh whole-domain and start/middle/end maps and return to the same visual review. A nonpass third attempt, unusable or stale evidence, or an unresolved judgment ends the repair loop by accepting the latest valid geometry with warnings. A fourth repair attempt remains forbidden.

## Feature and Map Policy

Every run derives target feature boxes before fitting the polygon. Required features may include upstream rivers, estuary/channel connectivity, forcing aprons, lake connections, island chains, and explicitly requested geopolitical or mission context.

The built-in regional feature-plan catalog is explicit heuristic guidance for the feature inference/scoring modules. In a Skill X-Ray, represent the catalog as one example leaf attached to T2 rather than one leaf per bounding box. Catalog boxes remain initial feature-plan seeds and never replace required-feature scoring or visual review.

Unknown or memory-disabled requests must never fall back to Delaware or another known box. Resolve them in this order: explicit `target_region_features` or polygon seed, catalog memory when enabled, cached online named-place discovery, then an agent-supplied researched or inferred `--discovery-bbox`. If all routes fail, return a nonzero `region_discovery_failed` error without writing a final `region_bpoly.json`.

Nominatim discovery is a bounded, user-triggered lookup: extract a concise named place from the modeling objective, issue at most one query, use an identifying User-Agent, cache the result, retain OpenStreetMap attribution, and record how the result was selected. A point-like result is expanded to a regional initial frame; it is not treated as authoritative mission coverage.

For every discovery-seeded coastal case, inspect the initial whole-domain map before final review. Confirm the scientific scope and explicitly correct `--offshore-azimuth-deg` when the inferred/default side is not ocean-facing. Then apply the normal hash-bound land-side truncation gate. Do not accept a discovered bbox merely because it geocoded successfully.

Use road-detail maps for small estuaries and topographic context for regional, lake, island-chain, and archipelago cases. Antimeridian domains require a compact longitude display frame assembled from separate native tile/coastline requests on each side of the dateline. The combined background must cover the complete display frame before it is considered geographically usable.

After coastal land-side review, tightness, truncation, obstruction, landing, and similar findings remain explicit nonblocking diagnostics. Invalid or absent four-corner geometry remains an execution error because there is no usable object to deliver.

## Domain Types

- `coastal`: one selected Atlantic/ocean-facing side for a later coastline-anchored OBC;
- `island`: offshore loop without mainland landfall anchors;
- `lake`: no ocean OBC;

The offshore reference point selects a side only. It is not an OBC endpoint or arc.

## Workflow States for X-Ray Casting

Preserve this operational ordering and loop:

- W1 interpret request
- W2 resolve heuristic mode
- W3 derive target features from explicit input, catalog memory, or bounded named-place discovery
- G1 usable initial geographic seed exists
  - absent catalog/explicit geometry: discover or supply an initial bbox and re-evaluate G1
  - all discovery routes fail: stop with `region_discovery_failed` and no delivered RegionBPoly
- W4 seed four-corner geometry
- W5 score/refit required coverage
- G2 required features and valid geometry
- W6 select domain type and offshore side
- W7 render initial/candidate evidence
- W8 write provisional candidate and hash bindings
- W9 inspect whole-domain and land-side start/middle/end maps
- G3 land-side truncation review
  - expand: W11 apply one authorized `expand_side`, then return to W9 with fresh maps
  - pass: W10 retain resolved QA, then W12 package standardized final evidence and provenance, then W13 terminal delivery
  - unresolved, unusable evidence, or attempt limit: W10 retain warning QA, then W12 package the latest valid best-effort geometry, then W13 terminal delivery

The five primary tool nodes are:

- T1 `run_region_bpoly.py`
- T2 feature inference/scoring modules
- T3 map rendering modules
- T4 `review_region_bpoly.py`
- T5 `adjust_region_bpoly.py`

## Outputs

A run that reaches W12 uses the canonical package defined in
[region_bpoly_output_contract.md](references/region_bpoly_output_contract.md).
The standardized root-level files are:

- `region_bpoly.json`
- `target_region_features.json`
- `region_bpoly_final_map.png`
- `offshore_boundary_artifacts.json`
- `region_bpoly_manifest.json`
- `region_bpoly_land_side_review.json` for finalized coastal domains, plus `.png` when the source maps were usable enough to compose it
- `region_place_discovery.json` when catalog/explicit feature geometry was unavailable

Every feature records its purpose, source kind, source key, and geometry status. The manifest records package state, delivery readiness, file sizes, and SHA-256 hashes. This packaging is part of W12 and does not introduce a new gate.

The final coastal JSON retains the review decision, iteration, side evidence, source hashes, validation findings, and compact-map path when available. A clean review records `qa.land_side_visual_gate.status: pass`; a best-effort delivery records `status: warning`. Both use `final_status: pass`. Consumers must treat review status as provenance rather than an intake gate and use the canonical filenames; `<name>_region_bpoly.json` may remain only as a compatibility alias.

## Validation

```powershell
python scripts/selftest_region_bpoly.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
