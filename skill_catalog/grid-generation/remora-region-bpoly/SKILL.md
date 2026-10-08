---
name: remora-region-bpoly
description: Define and visually refine four-sided REMORA modeling regions and related nesting or refinement areas, returning geographic polygons and a reviewed region plan independently of numerical grid generation.
---

# REMORA RegionBPoly

Translate the scientific request into geographic regions. Keep case data and maps in the user's REMORA workspace. This skill defines geographic intent; `remora-grid-generation` owns numerical footprints, staggering, masks, and eventual refinement alignment.

## Workflow

Use Python 3.11+ with binary packages from `requirements.txt`; install using `python -m pip install --only-binary=:all: -r requirements.txt` in a workspace environment.

1. Identify required physical features and the geographic target. Use explicit coordinates, supplied polygons, verified geography, or researched place discovery. Catalog boxes are seeds; never substitute an unrelated known region for an unresolved place.
2. Write a request using [the region interface](references/interface.md). Each region has a stable ID, scientific purpose, four vertices, required features, and optional parent. Preserve geographic source links. Do not impose FVCOM's single offshore-side convention.
3. Obtain a real coastline backdrop with `$gshhs-coastline` (estimate/cache-first). Record its source manifest and bbox. Run `python scripts/remora_regions.py build --request request.json --output-dir regions`.
4. **Open and inspect every map listed in `region_delivery.json`.** Inspect the whole domain, feature closeups, and parent-child overlays. Check feature coverage, entrance/channel placement, island cuts, and geographic suitability. Rendering alone is not review.
5. If necessary, edit vertices, features, or relationships and build a new output directory. Preserve the preceding attempt and explain physical reasons for changes.
6. Record actual inspection: `python scripts/remora_regions.py review --delivery regions/region_delivery.json --decision accepted --rationale "..."`. Use `needs_revision` if unsuitable. Changed evidence invalidates review.
7. Return `region_delivery.json`. Run `validate --delivery ... --require-reviewed` before downstream use. Several related regions do not imply generated nested grids.

## Boundaries

- v1 supports compact, non-polar, non-antimeridian four-sided regions. Report unsupported geography instead of distorting it.
- Child regions must reference an existing parent, have no relationship cycles, and lie inside that parent. Their role is scientific intent, not AMR indices.
- Region polygons do not become land masks. Open-edge conditions and numerical footprint fitting belong downstream.
- Use agent review without a routine human approval gate. Explain unresolved physical compromises.

The workflow adapts the feature-first, map-review, and frozen-delivery methods of `fvcom-region-bpoly`; see [provenance](references/provenance.md). No installed FVCOM files are modified or imported.
