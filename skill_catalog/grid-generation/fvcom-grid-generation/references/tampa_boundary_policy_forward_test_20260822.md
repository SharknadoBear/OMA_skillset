# Tampa Boundary-Policy Forward Test - 2026-08-22

A clean Codex agent forward-tested installed implementation commit `48f6901`
without inspecting or reusing earlier Tampa project artifacts. Large products
remain in the preprocessing workspace; paths below are relative to the fresh
standard project root.

## Boundary-policy result

- Placement policy: `offshore-first`.
- Selected family: `complete-offshore`.
- OBC: one simple, nonbranching 43-node Gulf arc with exactly two landfalls,
  no nonendpoint land crossing, and 138.319 km delivered length.
- Non-OBC open-water residual: 0.0 m.
- Residual fraction: 0.0.
- Coastline-plus-OBC exterior coverage: 1.0.
- All three independent hard gates passed; `report_only=false` and
  `downstream_eligible=true`.
- The hash-bound Codex whole-domain map decision passed and found no artificial
  frame-supported strip.
- Fresh post-arc bathymetry had 100% finite coverage. Its mixed-source vertical
  datum was not harmonized and remains an explicit scientific caveat.

The selected contract is under `03_boundary/`, and the project validator
rechecked its contract, decision, map, lineage, and selected-stage hashes.

## Terminal mesh and delivery result

| Metric | Result |
|---|---:|
| Nodes / triangles | 23,364 / 37,409 |
| Mesh SHA-256 | `5dd54e07a15bbf7975b68eb7b2051f81a9a26e4ae826f84205ecf14fbbb43e96` |
| q_min | 0.035746 |
| q_L3sigma | 0.463029 |
| Angle range | 1.188-172.781 deg |
| Maximum adjacent-area change | 0.963997 |
| Superthin elements | 27 |
| Maximum valence / nodes above 8 | 14 / 111 |
| Singly connected triangles | 66 |
| Wet components / nonmanifold edges | 1 / 0 |
| Positive finite depths / exact 2DM roundtrip | yes / yes |

The standardized delivery is complete at `final/fvcom_grid.2dm`, even though
the mesh is intentionally classified `needs_review`, `fvcom_ready=false`, and
`submission_eligible=false`. Ordinary project validation passes; validation
with `--require-submission-ready` fails only with
`project_not_submission_ready`. This demonstrates that the stable filename is
portable and predictable without implying scientific readiness.

Compact evidence is in `08_audit/final_audit.json` and
`final/fvcom_grid_status.json`. Remaining failure taxonomy is superthin debt,
low q_L3sigma, angle tails, adjacent-area transition, valence, and singly
connected elements.
