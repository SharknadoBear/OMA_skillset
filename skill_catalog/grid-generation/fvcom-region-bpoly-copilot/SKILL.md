---
name: fvcom-region-bpoly-copilot
harness: github-copilot
description: GitHub Copilot variant for creating FVCOM four-sided polygon domain envelopes with visual QA. Use run_in_terminal for script execution, view_image for map QA, vscode_askQuestions for review decisions. Scripts shared with sibling fvcom-region-bpoly/ folder.
---

# FVCOM Region BPoly — GitHub Copilot Harness

## Purpose

Turn a natural-language FVCOM modeling request into a QA-ready `RegionBPoly`: ordered four-corner polygon box with edge labels, domain type, open-boundary expectation, and visual QA artifacts.

## Scripts Location

All scripts in sibling folder:

```
Agent_skill_dev/skill_catalog/grid-generation/fvcom-region-bpoly/scripts/
```

Key scripts:
- `run_region_bpoly.py` — Primary streamlined workflow (prefer this)
- `propose_region_bpoly.py` — Generate candidate maps
- `classify_region_bpoly_domain.py` — Domain-type classification
- `review_region_bpoly.py` — Visual QA gate → accepted output
- `selftest_region_bpoly.py` — Static workflow checks

## Core Rule

The four-sided polygon (`polygon_lonlat`) is the controlling geometry. The `envelope_bbox` is only a fetch/plot helper.

## Copilot Execution Workflow

### Default: Execute Mode

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\grid-generation\fvcom-region-bpoly\scripts\run_region_bpoly.py" --request-text "Puget Sound tidal energy model" --run-dir Workspace/Preprocessing/fvcom-region-bpoly/runs/case --name case --mode execute
```

On success: outputs `region_bpoly.json` + `region_bpoly_final_map.png`, deletes intermediates.
On failure: keeps `intermediate/`, marks `final_status = needs_review`.

### Visual QA Gate (Copilot-Specific)

When the workflow produces candidate maps:

1. Use `view_image` to inspect `*_candidate_map.png` and `*_candidate_focus_map.png`
2. Use `read_file` on `*_candidate_score.json` for coverage metrics
3. Use `vscode_askQuestions` to ask the user for pass/revise/fail decision
4. If manual review needed, run:

```powershell
python "...\scripts\review_region_bpoly.py" --candidate-json runs/case/intermediate/case_region_bpoly_candidate.json --decision pass --domain-type-note-json runs/case/intermediate/case_domain_type_note.json --side-review-all-pass --notes "Visually verified in Copilot"
```

### Test Mode (Keeps Intermediates)

```powershell
python "...\scripts\run_region_bpoly.py" --request-text "..." --run-dir runs/case --name case --mode test
```

## Domain Types

- `coastal`: one ocean/open-boundary arc with two land anchors; needs open-boundary reference point
- `island`: offshore loop without land anchors
- `lake`: no ocean open boundary
- `unresolved_autonomous_failure`: not accepted

## Mission-Scope Gates

- `all tidal channels`, `tidal energy`, `connectivity` → connected straits/channels required
- Puget tidal-energy → Salish Sea scale (Puget Sound + Hood Canal + Admiralty Inlet + San Juan Islands + straits)
- Aleutian chain → must include intended extent or name subregion

## Place-Name Rules

- `Hawaii Island` = Big Island only
- `Hawaiian Islands`/`Hawaii state` = full chain
- Final map must show whole bpoly; antimeridian cases warn downstream

## Output Contract

Pass downstream only:
- `region_bpoly.json` — vertices, bbox, domain type, boundary policy, QA summary
- `region_bpoly_final_map.png`

## Copilot Tool Integration

| Step | Tool |
|------|------|
| Run scripts | `run_in_terminal` (mode=sync) |
| View candidate maps | `view_image` |
| Read score/coverage JSON | `read_file` |
| Ask user for QA decision | `vscode_askQuestions` |
| View final map | `view_image` on `region_bpoly_final_map.png` |
