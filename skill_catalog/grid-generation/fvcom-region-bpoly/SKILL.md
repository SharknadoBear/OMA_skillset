---
name: fvcom-region-bpoly
description: Create and iteratively review four-sided deformable polygon boxes for FVCOM preprocessing, including domain-type notes, mission-scope gates, map-visibility checks, derived fetch envelopes, and open-boundary expectations. Use when Codex needs to convert a natural-language coastal, island, lake, strait, or archipelago model-domain request into a QA-ready polygon domain envelope for downstream bathymetry, coastline, forcing, and grid-generation workflows.
---

# fvcom-region-bpoly

Use this skill to turn a natural-language FVCOM modeling request into a QA-ready `RegionBPoly`: an ordered four-corner polygon box with edge labels, derived fetch envelope, domain type, open-boundary expectation, and visual QA artifacts.

## Core Rule

The four-sided polygon is the controlling domain envelope. The axis-aligned `envelope_bbox` is only a helper for bathymetry/coastline fetching, source-coverage checks, and plot limits.

## Workflow

Default to execute mode unless the user explicitly asks for testing, benchmarking, debugging, or retained intermediate artifacts.

Run `scripts/run_region_bpoly.py`:

- `--mode execute` is the default. It writes one downstream JSON, `region_bpoly.json`, and one final map, `region_bpoly_final_map.png`. It deletes `intermediate/` after a successful pass.
- `--mode test` keeps the comprehensive `intermediate/` folder with candidate maps, side-focus maps, score JSON, and coverage JSON.
- If execute mode cannot pass, it keeps `intermediate/` and marks `final_status` as `needs_review`.

Do not generate meshes in this skill.

## RegionBPoly Object

`RegionBPoly` records:

- `polygon_lonlat`: ordered four-corner lon/lat polygon, closed in output;
- `edge_labels`: four edge names for visual review and downstream boundary logic;
- `offshore_azimuth_deg` and selected offshore side;
- derived `center_lon`, `center_lat`, `length_km`, `width_km`, `orientation_deg`;
- derived `envelope_bbox` for data fetching only;
- map-visibility and antimeridian warnings.

Keep four sides for v1. Consider five- or six-sided polygons only if repeated bpoly smoke tests remain below the desired 80-90 percent pass rate.

## Deformation Guidance

Prefer deforming individual vertices or edges over expanding the whole frame:

- stretch one side to include connected waterways;
- skew one corner to avoid cutting an island/headland;
- widen only the open-ocean side for forcing apron;
- preserve exactly one intended coastal open-boundary arc for coastal domains.

## Fast And Full Review

Default fast review writes an overview, one focus map, and four zooms along the inferred open/risky side.

Use `--full-side-review` when the region is complex, a side may cut connected water, an island may split the open boundary, the prompt asks for whole systems/all tidal channels/island chains, or the first candidate fails visual QA.

## Domain Types

Every accepted bpoly needs a domain-type note:

- `coastal`: one intended ocean/open-boundary arc with two land anchors; requires one approximate open-boundary reference point snapped to a bpoly edge.
- `island`: offshore loop/arc without land anchors.
- `lake`: no ocean open boundary.
- `unresolved_autonomous_failure`: not accepted.

## Mission-Scope Gates

The scientific mission controls required geography:

- `all tidal channels`, `tidal energy`, `connectivity`, `connected`, `whole system`, or `wave climate` make connected straits/channels required unless explicitly scoped out.
- Puget tidal-energy/all-channel prompts default to Salish Sea scale, including Puget Sound, Hood Canal, Admiralty Inlet, San Juan Islands, Haro/Boundary Pass, Strait of Georgia, and Strait of Juan de Fuca.
- Mission-critical connected water cannot be passed by labeling it optional.
- Aleutian island-chain prompts must include the intended chain extent or clearly name a scoped subregion.

## Place-Name And Visibility Gates

- `Hawaii Island` means Big Island unless the prompt says `Hawaiian Islands`, `Hawaii islands`, or `Hawaii state`.
- `Hawaiian Islands`, `Hawaii islands`, or `Hawaii state` means the island chain/state.
- The final map must visibly show the whole bpoly. Antimeridian cases must warn downstream tools to use `polygon_lonlat` / `RegionBPoly`, not a naive bbox.

## Scripts

- `scripts/run_region_bpoly.py`: primary streamlined workflow. Prefer this for new work.
- `scripts/propose_region_bpoly.py`: request/direct polygon -> bpoly candidate maps and JSON.
- `scripts/classify_region_bpoly_domain.py`: bpoly candidate -> domain-type note.
- `scripts/review_region_bpoly.py`: visual QA -> accepted `region_bpoly.json`.
- `scripts/benchmark_region_bpoly.py`: optional reference `.2dm` bbox comparison only.
- `scripts/selftest_region_bpoly.py`: static workflow checks.

Lower-level propose/classify/review scripts are for test mode, debugging, or manual QA. Legacy `*_bbox.py` scripts may exist only for compatibility. Prefer `run_region_bpoly.py` for all new work.

## Pass Requirements

Final pass requires:

- final map generated and visible for agent QA;
- all required ingredients inside the bpoly;
- required side-focus evidence in test/intermediate artifacts;
- mission-scope requirements satisfied by included ingredients;
- coastal domain has a domain-type note with open-boundary reference;
- single-open-boundary policy documented where applicable.

## Output Contract

Pass downstream only:

- `region_bpoly.json`
- `region_bpoly_final_map.png`

The final JSON contains polygon vertices, derived `envelope_bbox`, domain type, boundary policy, open-boundary reference, final map path, QA summary, and downstream-use notes.
