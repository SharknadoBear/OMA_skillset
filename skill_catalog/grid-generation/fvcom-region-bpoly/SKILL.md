---
name: fvcom-region-bpoly
description: Create and freely refine map-guided, feature-first four-sided RegionBPoly mission envelopes for FVCOM preprocessing, using coarse scientific review while always returning the latest valid geometry with explicit warnings.
---

# FVCOM RegionBPoly

Turn a regional ocean, estuary, lake, island, strait, or archipelago modeling objective into a QA-ready four-corner RegionBPoly.

Do not generate coastline boundary arcs, bathymetry, or meshes.

## Ownership and invariants

RegionBPoly owns mission scope, required-feature coverage, offshore-side orientation, and map-guided geographic suitability. Downstream skills must not request or apply RegionBPoly repair.

Preserve only the invariants needed by downstream consumers:

- exactly four finite vertices forming a valid, non-self-intersecting polygon;
- coverage of required mission features;
- usable geographic maps for scientific judgment;
- a standardized delivery package.

The delivered offshore OBC is not contained by RegionBPoly. It may deform outside the selected offshore side. envelope_bbox is a RegionBPoly-stage coastline-source and plotting helper, not a downstream bathymetry extent.

## Primary workflow

Run:

    python scripts/run_region_bpoly.py --request-text "..." --run-dir runs/case --name case --mode execute

For a forward test, use a new UTC-stamped directory under Workspace/Preprocessing/fvcom-region-bpoly/runs/; do not reuse earlier case artifacts. Use --mode test --heuristic-mode memory --basemap-provider auto when complete visual evidence must be retained.

- --heuristic-mode auto uses catalog memory in execute mode and bypasses it in test mode.
- With no explicit or catalog geometry, `--place-discovery auto` performs one cached discovery operation and uses the result only as an initial visual seed. Extract the geographic target before scientific-purpose clauses such as `to simulate` or `to study`. If a compound waterbody name has no combined result, expand shared plural types (for example, `A and B Bays`) into component queries, require every component to resolve, union their raw extents, and record all attempts and selected components.
- If lookup is unavailable or ambiguous, research or infer an initial frame and provide --discovery-bbox WEST SOUTH EAST NORTH with --discovery-label.
- --basemap-provider auto is the normal policy. none/off still requires a real offline coastline.
- CLI map generation is noninteractive and must force Matplotlib's `Agg` backend before importing `pyplot`; online-tile work must not create Tk objects or GUI-thread cleanup failures.
- Unknown named regions enter discovery; they never fall back to Delaware or another unrelated box.

A coastal run produces final_status: review_pending with a hash-bound region_bpoly_scientific_review_request_v1. Never stop at review_pending, repair_required, or needs_review. Continue scientific review and any useful repair. Once valid geometry exists, finish with final_status: pass, either as a scientifically accepted delivery or accepted_best_effort with explicit warnings.

Lake and island candidates may also be freely adjusted when their maps show an incomplete or scientifically unsuitable frame; they do not require the coastal hash-bound finalizer.

## Scientific review

Inspect the whole-domain map. For coastal domains also inspect the start, middle, and end context maps for every non-offshore side.

Use geographic and modeling judgment. Relevant problems can include:

- truncating a river, estuary arm, connected embayment, strait, or mission feature;
- a land side crossing mapped water where that crossing is scientifically inappropriate;
- bisecting an island or omitting part of a requested island chain;
- an offshore side or overall frame unsuitable for the requested circulation, mixing, exchange, salinity, or flooding objective.

These are considerations, not independent fields or deterministic side gates. Ordinary inland streams, incidental map detail, and harmless redundancy do not automatically require repair.

Finalize a suitable initial candidate with:

    python scripts/review_region_bpoly.py --candidate-json runs/case/region_bpoly.json --problem-detected no --problem-description "The initial frame covers the required system and has scientifically appropriate land and offshore placement." --change-required no --geometry-changed no --scientifically-useful yes --scientific-rationale "The mapped frame is suitable for the stated modeling objective." --map-visibility-status pass

The final review records only:

1. whether the agent recognized a meaningful problem or explicitly found none;
2. whether geometry changed when repair was required, verified by before/current hashes;
3. whether the final RegionBPoly is scientifically useful.

Do not grade the chosen operation, side, distance, orientation, or number of cycles.

## Agent-directed repair

When a meaningful problem exists, record a repair request:

    python scripts/review_region_bpoly.py --candidate-json runs/case/region_bpoly.json --problem-detected yes --problem-description "..." --change-required yes --geometry-changed no --change-description "..." --scientifically-useful no --scientific-rationale "The current frame is not yet suitable." --map-visibility-status pass

Choose any scientifically defensible repair. You may rotate, enlarge, contract, reshape vertices, move one or several sides including the offshore side, change offshore orientation, combine operations, or regenerate the candidate from a revised feature plan or seed. No operation or side is authorized in advance.

An adjustment manifest may contain one operation or an operations list:

    python scripts/adjust_region_bpoly.py --input-json runs/case/region_bpoly.json --adjustment-manifest runs/case/repair.json --output-json runs/case/region_bpoly_adjusted.json --map-path runs/case/region_bpoly_adjustment_map.png --repair-cycle

Rerender with any positive descriptive cycle number:

    python scripts/run_region_bpoly.py --request-text "..." --input-region-json runs/case/region_bpoly_adjusted.json --review-iteration 2 --run-dir runs/case_i02 --name case_i02 --mode test --heuristic-mode memory --basemap-provider auto

On acceptance after repair, provide --before-region-json and report --geometry-changed yes; the finalizer verifies the hash change. There is no numeric iteration limit. Continue while a meaningful scientific change remains.

If no useful repair remains, use --no-meaningful-repair-remaining. The latest valid geometry is delivered as accepted_best_effort with scientifically_useful: false; it is never mislabeled as scientifically accepted and never ends as needs_review.

## Feature and map policy

Derive target feature boxes before fitting the polygon. Required features can include upstream rivers, estuary/channel connectivity, forcing aprons, lake connections, island chains, and explicitly requested mission context.

Catalog and discovered boxes are initial seeds, not geographic truth. A discovery-seeded coastal case must verify scope and offshore orientation on the map. Antimeridian domains require compact longitude display assembled from separate native requests on each side of the dateline; reject smeared or incomplete backgrounds as scientifically usable evidence.

Domain types:

- coastal: one selected ocean-facing side for a later coastline-anchored OBC;
- island: offshore loop without mainland landfall anchors;
- lake: no ocean OBC.

## Workflow states for X-Ray casting

- W1 interpret objective
- W2 resolve heuristic mode
- W3 derive features from explicit input, catalog memory, or named-place discovery
- G1 usable initial geographic seed
- W4 seed four-corner geometry
- W5 score/refit required coverage
- G2 required features and valid geometry
- W6 select domain type and offshore side
- W7 render candidate evidence
- W8 write provisional candidate and hash bindings
- W9 inspect scientific suitability
- G3 coarse scientific review
  - repair: W11 apply any agent-selected geometry change, rerender, and return to W9
  - accept: W10 retain review provenance, W12 package standardized evidence, W13 terminal delivery
  - no meaningful repair remains: W10 retain warning provenance, W12 package latest valid best effort, W13 terminal delivery

Primary tools:

- T1 run_region_bpoly.py
- T2 feature inference/scoring modules
- T3 map rendering modules
- T4 review_region_bpoly.py
- T5 adjust_region_bpoly.py

## Outputs

Use [region_bpoly_output_contract.md](references/region_bpoly_output_contract.md). Canonical root files are:

- region_bpoly.json
- target_region_features.json
- region_bpoly_final_map.png
- offshore_boundary_artifacts.json
- region_bpoly_manifest.json
- region_bpoly_land_side_review.json for finalized coastal domains, plus its compact map when usable
- region_place_discovery.json when named-place discovery was used

Clean and best-effort deliveries both use final_status: pass; the review outcome and scientifically_useful field distinguish them. Downstream consumers treat review provenance as nonblocking and consume the latest valid geometry.

## Validation

    python scripts/selftest_region_bpoly.py
    python -m compileall scripts
    python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
