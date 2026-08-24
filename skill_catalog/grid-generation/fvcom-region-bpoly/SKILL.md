---
name: fvcom-region-bpoly
description: Create and geometrically adjust map-guided, feature-first four-sided RegionBPoly envelopes for FVCOM preprocessing, including post-arc GSHHS clipping feedback, target-feature preservation, offshore-side artifacts, and QA scoring.
---

# fvcom-region-bpoly

Use this skill to turn a natural-language FVCOM regional ocean, estuary, lake, island, strait, or archipelago modeling request into a QA-ready `RegionBPoly`: an ordered four-corner polygon box with feature coverage, domain type, offshore-side intent, final map, and review artifacts.

## Core Rule

The four-sided `polygon_lonlat` is the controlling model-domain envelope. The `envelope_bbox` is only a helper for bathymetry/coastline fetching, source-coverage checks, and plot limits.

Do not generate meshes or coastline boundary arcs in this skill.

GSHHS remains downstream in `fvcom-bdry-arc`. This skill may consume its current `region_bpoly_arc_feedback_v2` artifact (or historical v1 evidence) to reshape the four-corner polygon, but it must not fetch or interpret coastline data itself.

## Offshore Point Purpose

The offshore point only identifies the intended offshore side for downstream coastline-anchor snapping. Downstream tools can use that side to snap onto the two outermost coastline anchor points and decide where the offshore boundary arc belongs.

This skill does not generate that boundary arc.

## Primary Workflow

Run `scripts/run_region_bpoly.py` by default:

- `--mode execute`: final-only downstream output. On pass, writes `region_bpoly.json`, `region_bpoly_final_map.png`, and `offshore_boundary_artifacts.json`, then removes `intermediate/`.
- `--mode test`: writes the same final outputs and retains `intermediate/visual_review/` with initial guess maps, feature plans, feature GeoJSON, candidate maps, side zoom maps, score JSON, and coverage JSON.
- `--review-depth auto|fast|full` defaults to `auto`.
- `--full-side-review` remains a backward-compatible alias for `--review-depth full`.
- `--heuristic-mode auto|memory|unknown` defaults to `auto`: execute resolves to `memory`; test resolves to `unknown`.
- `--basemap-provider auto` is the default. Every initial, candidate, focus, side-review, adjustment, and final map must include usable geographic background context. A flat water fill or coordinate grid is diagnostic only and must yield `final_status=needs_review`; it is never sufficient for RegionBPoly acceptance.
- `--offshore-azimuth-deg` can override the selected offshore side after candidate repair when map review requires a different side selector.

If execute mode cannot pass, it keeps `intermediate/` and marks `final_status` as `needs_review`.

Prefer `--mode test` for benchmarks, subagent tests, debugging, or retained VLM/agent-review evidence.

## Memory And Unknown-Region Policy

Heuristics are practical workflow memory, not a scientific guarantee. In execute mode, place-memory heuristics, known-region feature libraries, deformation presets, and deterministic repair candidates remain enabled so the skill can handle known domains efficiently.

In test mode, `--heuristic-mode auto` resolves to memory-off `unknown` mode. This disables hard-coded place guesses, deformation presets, repair candidates, and feature-library inference unless the request supplies explicit `target_region_features`, explicit required ingredients, or an explicit polygon seed. The purpose is to expose the real uncertainty of a raw prompt during subagent screening.

Unknown regions must never fall back to Delaware/NJ or any other accepted default box. If a memory-off run lacks explicit feature geometry, write final artifacts with `final_status: needs_review`, `domain_type: unresolved_autonomous_failure`, and failure code `unknown_region_no_feature_plan`.

Explicit feature boxes remain valid in test mode. When the request supplies `target_region_features` or required feature ingredients, build the bpoly from those boxes without using place-memory shortcuts.

## Feature-First Planning

Every run decomposes the prompt into target-region feature boxes before scoring the bpoly:

- `target_region_features.json`: prompt-derived feature plan.
- `target_region_feature_polygons.geojson`: feature boxes used as coverage/scoring ingredients.

The feature plan must consider:

- forcing data sources and offshore forcing apron;
- offshore extension away from constricted inlets and broad-region island barriers;
- upstream river-input extent;
- estuary/channel connectivity;
- lake inlets, outlets, and connecting channels;
- complete island, island-chain, and archipelago inclusion;
- geopolitical separation when it can clip connected hydrodynamic context.

The initial guess map is generated before feature-envelope repair. If a bad initial guess misses required features, the refit must use normalized place-name and longitude-frame logic rather than letting the bad candidate contaminate the corrected extent.

Feature plans may include non-required `offshore_boundary_exclusion` / `obstruction_guard` boxes. These are not target coverage ingredients, but they are QA guards for islands or land masses that must not cut the selected offshore side.

## Background Maps

Never accept a blank lat/lon-only plot as visual evidence. `region_bpoly_final_map.png`, initial guess maps, candidate maps, focus maps, side zoom maps, and adjustment maps must include background geography.

With `--basemap-provider auto`, small-estuary and creek-scale cases use `road_detail`, while regional, lake, island-chain, and archipelago cases use topographic context. Do not override `auto` merely for deterministic execution. `road_detail` tries Esri World Street Map, CARTO Voyager, and OpenStreetMap Mapnik. Regional topographic context tries Esri World Topographic Map, OpenTopoMap, CARTO Voyager, and OpenStreetMap Mapnik. Both chains then use a local coastline fallback. Offline coastline discovery follows the project output path as well as the current process path so Windows short-drive launchers cannot hide the workspace cache, and prefers full-resolution GSHHS over high-resolution GSHHS and Natural Earth. Record the selected provider and every failed predecessor. `none/off/false` means skip online tiles and require a real offline coastline; if no coastline is found, the workflow retains diagnostic maps but cannot pass.

For small-estuary cases, side zoom maps use a smaller focus radius and explicit target zoom, normally 13 within the 13-15 inspection range, so the agent can inspect river mouth, tidal-creek, and immediate-bay geometry instead of a coarse regional frame. The resolved map-detail policy is recorded in final JSON and map metadata. In test mode, small-estuary cases write a `basemap_comparison/` folder with Esri Street, CARTO Voyager, OSM, topo, and offline fallback maps.

Antimeridian cases must be drawn in a compact longitude display frame so the map is regional and reviewable, not a world-spanning naive bbox.

## Review Depth

Fast review creates 4 zoom maps: one centered on each bpoly side.

Full review creates 12 zoom maps: start, middle, and end on all four sides.

`auto` selects full review for complex cases, including:

- archipelago or island-chain requests;
- all-channel, tidal-energy, connectivity, connected-water, or wave-climate missions;
- multiple river/channel/inlet contexts;
- lake connecting-channel cases;
- geopolitical split risks;
- antimeridian-crossing cases;
- failed required-feature coverage before refit.

All retained review maps live under `intermediate/visual_review/`.

## QA Scoring

Final JSON must include QA for:

- required feature coverage;
- tight feature fit;
- wrong-region inclusion risk;
- domain type;
- offshore-side quality;
- offshore-obstruction guards, including whether the selected side or domain intersects guard boxes;
- broad-region island crossing risk along the offshore side;
- coastal/estuary open-gate landing quality, including whether adjacent sides give downstream `fvcom-bdry-arc` a practical solid-coast landing;
- antimeridian safety and map display span;
- practical map usability.

Known scope rules:

- Cook Inlet wave, wave-current, SWAN, wave-climate, offshore-wave-forcing, or fetch prompts use `domain_variant: cook_inlet_wave_fetch`: include Kodiak Island as required context, include Augustine Island and Ursus Cove/Kamishak west-side Cook Inlet context, push the offshore side well south/offshore for wave fetch, and avoid unnecessary Prince William Sound overreach on the east side.
- Cook Inlet tidal-only/current-only/residual-transport prompts use `domain_variant: cook_inlet_tidal_mouth`: keep the tighter mouth-gate domain and avoid routing the offshore side through Kodiak Island, represented as an obstruction guard.
- Mobile Bay prompts must include Mobile Bay, the lower Mobile-Tensaw river/delta context, a Gulf-facing gate, and enough westward landing context beyond Horn Island for the downstream boundary arc. Avoid unnecessary Perdido Bay and Wolf Bay inclusion unless explicitly requested.
- Murderkill-style requests are `small_estuary` scale and must pass tighter area/width limits rather than broad Delaware-Bay-scale coverage.
- Hawaii Island / Big Island-only domains must avoid Maui Nui and neighboring-island obstruction guards; Hawaii State/island-chain domains may include the island chain.
- `Hawaii Island` means Big Island unless the prompt says `Hawaiian Islands`, `Hawaii islands`, or `Hawaii state`.
- `Hawaiian Islands`, `Hawaii islands`, or `Hawaii state` means the state/island chain.
- `South-east Alaska`, `southeast Alaska`, and related forms normalize to Southeast Alaska, not a continent-scale fallback.
- Puget/Salish connectivity missions are connected coastal/inland-sea domains, not island-domain policies.
- Lake domains have no ocean open-boundary reference.

## Domain Types

Every accepted bpoly needs a domain type in final JSON:

- `coastal`: one intended ocean/open-boundary side or arc with land anchors downstream; requires one snapped offshore-side reference point.
- `island`: offshore loop/arc without land anchors.
- `lake`: no ocean open boundary.
- `unresolved_autonomous_failure`: not accepted.

## Adjustment Tools

Use `scripts/adjust_region_bpoly.py` when the agent needs direct final-stage polygon edits from map review.

Use `scripts/apply_arc_feedback.py` for a geometry-only adjustment requested by `fvcom-bdry-arc`. The command defaults to the resilient `auto` basemap chain, verifies the feedback hash, applies one named full-edge or tapered reshape candidate, preserves `target_region_features` exactly by canonical hash, recomputes required-feature and obstruction QA, resnaps the offshore reference, and writes a complete downstream-compatible RegionBPoly plus a geography-usable comparison map. Reject stale feedback, semantic mutations, invalid polygons, lost required features, and new obstruction conflicts.

The adjustment manifest supports:

- `rotate`: rotate the four-corner bpoly around `center`, `offshore_midpoint`, or explicit `pivot_lonlat`.
- `scale`: enlarge/shrink along and across the polygon orientation with `factor`, or separate `along_factor` and `across_factor`.
- `reshape`: perturb individual vertices with `vertex_delta_km` east/north offsets.

The adjustment map must overlay the old polygon as a dashed line and the adjusted polygon as a solid line.

Before final pass, the workflow can test deterministic repair candidates generated from the same rotate/scale/reshape-style perturbation logic. A repair candidate may replace the current bpoly only when it preserves required-feature coverage, avoids obstruction guards, improves tightness or removes blocking failures, and remains a valid four-sided RegionBPoly. If no safe four-sided candidate exists, mark `final_status: needs_review`.

Example:

```powershell
python scripts/adjust_region_bpoly.py --input-json runs/case/region_bpoly.json --adjustment-manifest runs/case/adjust.json --output-json runs/case/region_bpoly_adjusted.json --map-path runs/case/region_bpoly_adjustment_map.png
```

## Output Contract

Pass downstream:

- `region_bpoly.json`
- `region_bpoly_final_map.png`
- `offshore_boundary_artifacts.json`

`offshore_boundary_artifacts.json` records selected side index/name, side endpoints, midpoint, snapped reference point, snap distance, offshore azimuth, boundary policy, review depth, zoom-review metadata, warnings, and failure taxonomy.

## Scripts

- `scripts/run_region_bpoly.py`: primary final-output workflow.
- `scripts/propose_region_bpoly.py`: request/direct polygon to feature plan, candidate maps, zoom maps, and JSON.
- `scripts/adjust_region_bpoly.py`: perturbative rotate/scale/reshape adjustment CLI with dashed-old/solid-new comparison map.
- `scripts/apply_arc_feedback.py`: apply one hash-bound, geometry-only boundary-arc feedback candidate and rerun RegionBPoly QA.
- `scripts/classify_region_bpoly_domain.py`: bpoly candidate to domain-type note.
- `scripts/review_region_bpoly.py`: visual QA to accepted `region_bpoly.json`.
- `scripts/selftest_region_bpoly.py`: static workflow and adjustment-tool checks.

Lower-level and legacy `*_bbox.py` scripts exist for compatibility or debugging. Prefer `run_region_bpoly.py` for new work.
