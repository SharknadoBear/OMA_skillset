---
name: remora-region-bpoly
description: Define and visually refine science-driven four-sided REMORA regions, feature priorities, boundary intents, and future nesting interests, returning geographic polygons and a reviewed plan independently of numerical grid generation.
---

# REMORA RegionBPoly

Translate the scientific request into geographic regions. Start from the intended dynamics and essential water connections, then decide the geographic extent. Keep case data and maps in the user's REMORA workspace. This skill defines geographic intent; `remora-grid-generation` owns the numerical footprint, resolution, staggering, masks, and eventual refinement alignment.

## Scientific decisions

Distinguish `barotropic_tide` from `baroclinic` purpose before numerical design. Preserve the user's scientific prompt as prose; numerical parameters belong in the downstream grid request. Use [scientific design guidance](references/scientific_design.md) when choosing extent, feature priorities, and boundary placement.

- Mark features `essential`, `optional`, or `future_refinement`. Essential entrances, islands, passages and water connections must survive the plan and subsequent grid. Optional simplification or exclusion needs source evidence and a physical rationale. Future refinement is an interest for later nesting, not a resolution claim for this root grid.
- Include enough geography to represent the requested dynamics. Do not automatically extend upstream, into peripheral bays, or into deep offshore water. Explain additional coverage in terms of boundary placement, transport pathways, or other relevant physics.
- Place open boundaries intentionally and avoid cutting islands or essential channels. Several open sides are valid. Geographic sides do not yet determine the numerical grid's logical boundary labels.
- Keep the root domain computationally proportionate. A tiny secondary feature does not by itself justify fine resolution everywhere; prioritize essential connections, supported local grading, evidence-backed optional simplification, or later nesting.

## Workflow

Use Python 3.11+ with binary packages from `requirements.txt`; install using `python -m pip install --only-binary=:all: -r requirements.txt` in a workspace environment.

1. Identify the purpose, geographic target, essential physical features and priorities. Use explicit coordinates, supplied polygons, verified geography, or researched place discovery. Catalog boxes are seeds; never substitute an unrelated known region for an unresolved place.
2. Write a request using [the region interface](references/interface.md). Each region has a stable ID, scientific purpose, four vertices, features, boundary intents when known, and an optional parent. Preserve geographic source links and evidence for compromises. Do not impose FVCOM's single offshore-side convention.
3. Obtain a real coastline backdrop with `$gshhs-coastline` (estimate/cache-first). Record its source manifest and bbox. Run `python scripts/remora_regions.py build --request request.json --output-dir regions`.
4. **Open and inspect every map listed in `region_delivery.json`.** Inspect the whole domain, feature closeups, and parent-child overlays against actual source geography. Check essential connections, proposed exclusions, entrance/channel placement, island cuts, boundary intent, and avoidable extra extent. Rendering alone is not review; a passing polygon containment test is not geographic acceptance.
5. If necessary, edit vertices, features, or relationships and build a new output directory. Preserve the preceding attempt and explain physical reasons for changes.
6. Record actual inspection: `python scripts/remora_regions.py review --delivery regions/region_delivery.json --decision accepted --rationale "..."`. Use `needs_revision` if unsuitable. Changed evidence invalidates review.
7. Return `region_delivery.json`. Run `validate --delivery ... --require-reviewed` before downstream use. Several related regions do not imply generated nested grids.

## Output ownership

Use the caller's designated directory exactly. In the standard REMORA case workflow, return outputs under `CASE/attempts/attempt_NNN/01_regions/` and keep the source request alongside the other attempt requests. Do not create an alternative project folder or write generated files inside installed skills. Geographic plans and their reviews remain independently hash-bound when the parent gridding skill assembles its final package.

## Boundaries

- The compatible v1 file schemas support compact, non-polar, non-antimeridian four-sided regions, including the optional v2 scientific metadata. Report unsupported geography instead of distorting it.
- Child regions must reference an existing parent, have no relationship cycles, and lie inside that parent. An `outer_domain` has no parent; `nesting_interest` and `refinement_interest` require one. Their role is scientific intent, not AMR indices. Unknown legacy role labels remain compatible.
- Region polygons do not become land masks. Open-edge conditions and numerical footprint fitting belong downstream.
- Use agent review without a routine human approval gate. Explain unresolved physical compromises.

The workflow adapts the feature-first, map-review, and frozen-delivery methods of `fvcom-region-bpoly`; see [provenance](references/provenance.md). No installed FVCOM files are modified or imported.

Run `python scripts/selftest.py --artifact-dir WORKSPACE/validation/region_tests` in a new output directory for priority, hierarchy, retention, and stale-source tests. Synthetic review receipts exercise hash binding; they do not constitute scientific visual review.
