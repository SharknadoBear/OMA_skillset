# Minimal-topology six-case evidence (2026-08-17)

This is the path-neutral evidence summary for campaign
`simplified_conditioning_six_case_20260817T023035Z`, run from the installed
skill on branch `feat/simplified-grid-conditioning-six-case` at implementation
commit `cef5558`. This opening section preserves the original campaign record.
Its workspace directory was later overwritten by the authorized three-case
gate-relaxation rerun documented below, so the old large artifacts are no
longer present; the committed hashes and summary remain historical evidence.

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

Delaware provisionally closed all 40 valence violations, and the historical
Hawaii row provisionally closed all 269. Both atomic batches were rolled back
under the then-current outer policy because the area-transition defect count
and the lower quality tail `q_p01` regressed. The Hawaii row used the Gmsh-5
rejection fixture from `final4_hawaii_g5_postbatch_20260730`; it is not the
correct Gmsh-6 Hawaii target and is superseded by the rerun below.

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

## Three-case outer-gate relaxation rerun

The same campaign directory was recreated from the active installed skill at
implementation commit `8cd8258`. For `minimal-topology-v1` only, outer-stage
changes in `q_p01` and the count of adjacent-area defects above `0.50` were
made report-only. Local transaction gates, structural invariants, maximum-area
jump, bounded quality floors, `q_L3sigma`, and `L/h` protections remained
active. Legacy profiles retained both former vetoes.

- Campaign runtime: 891.25 s.
- Campaign JSON SHA-256: `2e63a6d631191c8554d95c1761b82bad8b7eb5ce8c78f9e6d44f051edf11afdd`.
- Campaign CSV SHA-256: `14835bea30297978014a2bac3054e6b4bed5f837abf4faef5ac44a897de99dbd`.
- Report SHA-256: `bd39d371c6dd974d41767d0073719bcd0eef055cdae2dcd4afdcf3535f5b11bd`.
- Driver results: three attempted and three serialized; zero driver failures.
- Minimal local debt closure: one of three.
- Full FVCOM readiness: zero of three.

| Rank | Case | Nodes pre->post | Triangles pre->post | qL3sigma pre->post | Superthin pre->post | Valence >8 pre->post | Local closure | Runtime s |
|---:|---|---:|---:|---:|---:|---:|---|---:|
| 1 | Delaware Bay | 169,237->169,278 | 302,623->302,705 | 0.852419->0.852426 | 3->3 | 40->0 | no | 361.28 |
| 2 | San Francisco Bay | 44,639->44,640 | 84,250->84,252 | 0.875832->0.875899 | 1->1 | 1->0 | no | 52.61 |
| 3 | Hawaii Gmsh-6 | 344,941->344,956 | 657,667->657,697 | 0.887658->0.887628 | 0->0 | 16->0 | yes | 477.28 |

| Case | q_p01 delta | Area-defect count | Primary rollback |
|---|---:|---:|---|
| Delaware Bay | -0.000127686 | 1,482->1,487 | no |
| San Francisco Bay | +0.000041479 | 258->254 | no |
| Hawaii Gmsh-6 | -0.000038358 | 1,576->1,575 | no |

Delaware now retains the useful 40-to-zero valence repair that the previous
outer policy discarded. Its three existing superthin elements remain, so it
does not claim minimal closure. San Francisco remains a control: its one
valence violation closes, quality and area-defect count improve, but the one
protected superthin element remains. The correct Hawaii Gmsh-6 input closes
all 16 valence violations without creating superthin debt and reaches minimal
local closure. Its small `q_p01` and `q_L3sigma` decreases remain within the
revised report-only and retained tolerance contracts, respectively.

All three delivered meshes retain one wet component, zero nonmanifold edges,
exact fixed boundary geometry, ordered OBC lineage, finite positive-down
depths, and passing 2DM roundtrips. Full readiness remains false because of
pre-existing angle, adjacent-area, bathymetric-slope, singly connected,
boundary-continuity, and/or forcing debt. Hawaii additionally retains its
cyclic-OBC self-description limitation. No remaining outer gate rejected a
real-case candidate in this rerun; deterministic rejected-candidate retention
was instead covered by the synthetic installed-skill tests.
