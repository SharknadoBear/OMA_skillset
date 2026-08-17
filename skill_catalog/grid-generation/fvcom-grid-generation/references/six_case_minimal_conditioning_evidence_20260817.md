# Minimal-topology six-case evidence (2026-08-17)

This is the path-neutral evidence summary for campaign
`simplified_conditioning_six_case_20260817T023035Z`, run from the installed
skill on branch `feat/simplified-grid-conditioning-six-case` at implementation
commit `cef5558`. The full immutable evidence remains under
`Workspace/Preprocessing/fvcom-grid-generation/runs/` and is intentionally not
committed.

- Requested profile: `auto`; effective profile: `minimal-topology-v1`.
- Campaign runtime: 3,452.53 s.
- Campaign JSON SHA-256: `d37a628bd392de8083f5589238762c85271dccfb0ddbe36eafdacc7c9a416ae9`.
- Driver results: six attempted, five serialized, one input failure.
- Minimal local debt closure: two of six.
- Full FVCOM readiness: zero of six.
- No composite cross-region score was computed.

| Rank | Case | Nodes pre->post | Triangles pre->post | qL3sigma pre->post | Superthin pre->post | Valence >8 pre->post | Local closure | FVCOM ready | Runtime s |
|---:|---|---:|---:|---:|---:|---:|---|---|---:|
| 1 | Lake Superior | 41,398->41,398 | 77,983->77,983 | 0.89064->0.89064 | 0->0 | 0->0 | yes | no | 32.06 |
| 2 | Long Island Sound | 97,079->97,084 | 180,843->180,853 | 0.88910->0.88910 | 0->0 | 4->0 | yes | no | 93.37 |
| 3 | San Francisco Bay | 44,639->44,640 | 84,250->84,252 | 0.87583->0.87590 | 1->1 | 1->0 | no | no | 50.48 |
| 4 | Delaware Bay | 169,237->169,237 | 302,623->302,623 | 0.85242->0.85242 | 3->3 | 40->40 | no | no | 359.03 |
| 5 | Hawaii | 368,950->368,950 | 705,685->705,685 | 0.83500->0.83500 | 0->0 | 269->269 | no | no | 2,870.74 |
| 6 | Cook Inlet | 577,248->NA | 1,084,424->NA | 0.87101->NA | 43->NA | 20->NA | no | no | 46.17 |

Lake Superior exercised the deterministic no-op path. Long Island Sound closed
all four valence violations with four local cavity transactions and created no
superthin elements. San Francisco Bay closed its single valence violation, but
its severe superthin element had no protected-edge-safe repair that passed all
non-regression gates.

Delaware provisionally closed all 40 valence violations, and Hawaii
provisionally closed all 269. Both atomic batches were correctly rolled back
because the area-transition defect count and the lower quality tail `q_p01`
regressed. The final meshes therefore preserve their original topology debt.
Hawaii also retains its cyclic-sidecar and forcing-compatibility readiness
failures.

Cook Inlet stopped before any edit because two raw-mesh nodes were outside
finite positive bathymetry coverage; the first failing coordinate was
`(-152.4069756294, 58.6608315948)`. Its known incorrectly placed OBC remains a
separate scientific-input failure. No boundary, bathymetry, or raw-mesh source
artifact was changed.

For all five serialized cases, the final mesh retained one wet component, zero
nonmanifold edges, fixed boundary coordinates and membership, protected/OBC
lineage, finite positive depths, and an exact passing 2DM roundtrip audit. All
before/after diagnostic commands passed. Full readiness remained blocked by
pre-existing combinations of singly connected elements, angle debt,
area-transition debt, bathymetric-slope debt, boundary-continuity debt, and/or
OBC forcing limitations.

The evidence supports `minimal-topology-v1` as a useful no-op and small-valence
profile, but not yet as a general closure method for severe superthin debt or
large valence batches. The whole-batch rollback behavior is conservative and
scientifically safe, though Hawaii's runtime shows that future refinement
should consider smaller independently audited transaction clusters.
