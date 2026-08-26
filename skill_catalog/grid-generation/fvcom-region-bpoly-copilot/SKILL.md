---
name: fvcom-region-bpoly-copilot
harness: github-copilot
description: GitHub Copilot variant for creating map-guided, feature-first FVCOM four-sided RegionBPoly envelopes with background-map QA, adjustment tools, and offshore-boundary artifacts. Scripts are shared with sibling fvcom-region-bpoly/.
---

# FVCOM Region BPoly - GitHub Copilot Harness

## Purpose

Turn a natural-language FVCOM modeling request into a QA-ready `RegionBPoly`: ordered four-corner polygon box, target-region feature coverage, domain type, offshore-side intent, final background map, and visual QA artifacts.

All scripts live in the sibling folder:

```text
Agent_skill_dev/skill_catalog/grid-generation/fvcom-region-bpoly/scripts/
```

Prefer `run_region_bpoly.py` for new work.

## Core Rules

The four-sided `polygon_lonlat` is the controlling geometry. The `envelope_bbox` is only a fetch/plot helper.

The offshore point only identifies the intended offshore side for downstream coastline-anchor snapping. Boundary arc generation is outside this skill.

Every initial, candidate, zoom, adjustment, and final map must include background map context. `--basemap-provider auto` is the default: use `road_detail` for small-estuary and creek-scale cases, topographic context for regional/lake/island-chain cases, then local coastline/minimal fallback if online tiles are unavailable. `road_detail` tries Esri World Street Map, CARTO Voyager, then OpenStreetMap Mapnik with explicit small-estuary zoom policy. A `none/off/false` basemap request means offline fallback, not a blank lat/lon-only plot.

Use `--offshore-azimuth-deg` only when map review requires an explicit offshore-side selector override after the normal candidate repair pass.

## Execute Mode

```powershell
python "Agent_skill_dev\skill_catalog\grid-generation\fvcom-region-bpoly\scripts\run_region_bpoly.py" --request-text "Puget Sound tidal energy model" --run-dir Workspace/Preprocessing/fvcom-region-bpoly/runs/case --name case --mode execute
```

For every resolved proposal, execute mode outputs only:

- `region_bpoly.json`
- `region_bpoly_final_map.png`
- `offshore_boundary_artifacts.json`

It removes `intermediate/` after delivery. Late map, tightness, obstruction, landing, and related QA findings remain in `delivery_warnings` and do not stop automatic handoff.

## Test / Review Mode

```powershell
python "...\scripts\run_region_bpoly.py" --request-text "..." --run-dir runs/case --name case --mode test --review-depth auto
```

Test mode keeps `intermediate/visual_review/` with:

- initial guess map and JSON;
- `target_region_features.json`;
- `target_region_feature_polygons.geojson`;
- candidate and focus maps;
- side zoom maps;
- coverage and score JSON.

Use `--heuristic-mode auto|memory|unknown` for memory hygiene. `auto` resolves to memory-on in execute mode and memory-off `unknown` in test mode. In memory-off test mode, do not use hard-coded place guesses, feature-library inference, deformation presets, deterministic repair candidates, or Delaware/NJ fallback boxes. If no explicit `target_region_features`, required ingredients, or polygon seed are supplied, return `final_status: needs_review` with `unknown_region_no_feature_plan`. Explicit feature boxes remain valid in test mode.

## Review Depth

- `--review-depth fast`: 4 zoom maps, one centered on each side.
- `--review-depth full`: 12 zoom maps, start/middle/end on all sides.
- `--review-depth auto`: full for complex archipelago, all-channel, connectivity, multiple river/channel, lake-connection, geopolitical, antimeridian, or failed-coverage cases.
- `--full-side-review`: backward-compatible alias for full review.

## Visual QA Diagnostics

When candidate maps are retained:

1. Use `view_image` on initial guess, candidate, focus, final, and relevant side zoom maps.
2. Use `read_file` on score JSON, `target_region_features.json`, `region_bpoly.json`, and `offshore_boundary_artifacts.json`.
3. Use `vscode_askQuestions` for pass/revise/fail or targeted side-boundary decisions when needed.

Record a delivery warning when map metadata shows no enabled background context. The plotting layer should report `enabled: true` and `required: true`, using Esri topo tiles, OSM road tiles, an offline coastline background, or the minimal geographic fallback, but missing context does not stop a resolved polygon from automatic delivery.

## QA Expectations

- Required feature boxes must be scored.
- The box should fit required features tightly and avoid wrong-region inclusion.
- Non-required `offshore_boundary_exclusion` / `obstruction_guard` boxes must be used to prevent false passes when the offshore side is cut by blocker islands.
- Cook Inlet wave, wave-current, SWAN, wave-climate, offshore-wave-forcing, or fetch prompts use `cook_inlet_wave_fetch` and must include Kodiak, Augustine Island, Ursus Cove/Kamishak, and a broad Gulf wave apron while avoiding unnecessary Prince William Sound overreach.
- Cook Inlet tidal-only/current-only prompts use `cook_inlet_tidal_mouth` and must avoid Kodiak as an obstruction guard.
- Mobile Bay domains must include the Mobile-Tensaw delta context and keep the Gulf gate west enough to land beyond Horn Island while avoiding unnecessary Perdido Bay / Wolf Bay inclusion.
- Murderkill-style prompts are `small_estuary` scale and must satisfy tighter area/width limits with `road_detail` map policy.
- Hawaii Island / Big-Island-only domains must avoid Maui Nui / neighboring-island obstruction guards; Hawaii State/island-chain requests may include the island chain.
- Southeast Alaska name variants must not fall back to a continent-scale box.
- Puget/Salish connectivity missions are connected coastal/inland-sea domains, not island-domain policies.
- Aleutian/antimeridian maps must be compact and reviewable.
- Lake domains have no ocean open-boundary reference.
- Hawaii Island means Big Island unless the prompt asks for all Hawaii state/island-chain context.

## Adjustment Workflow

Use `adjust_region_bpoly.py` when direct map-based polygon edits are needed.

The workflow may also test deterministic four-sided repair candidates before delivery. Use a repair only if it preserves required-feature coverage, avoids obstruction guards, improves tightness or removes reported failures, and remains a valid RegionBPoly. Otherwise retain the current resolved polygon, record the remaining QA findings, and continue downstream.

Supported manifest operations:

- `rotate`
- `scale`
- `reshape`

The adjustment map must show old polygon dashed and adjusted polygon solid.

```powershell
python "...\scripts\adjust_region_bpoly.py" --input-json runs/case/region_bpoly.json --adjustment-manifest runs/case/adjust.json --output-json runs/case/region_bpoly_adjusted.json --map-path runs/case/region_bpoly_adjustment_map.png
```

## Output Contract

Pass downstream:

- `region_bpoly.json`
- `region_bpoly_final_map.png`
- `offshore_boundary_artifacts.json`

The final PNG should include background map geography, not only the polygon and grid. If geography is unavailable, retain that fact in `delivery_warnings` rather than creating a late delivery stop.
