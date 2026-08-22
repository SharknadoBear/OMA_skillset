# Autonomous Thin-Boundary Proof — 2026-08-22

Two pre-existing Gmsh-6 cases exercised the experimental workflow before the
fresh-estuary forward test. Large artifacts remain outside the catalog.

| Case | Classification | Superthin before → after | q_L3σ before → after | Final nodes / triangles | Thin closed | Minimal debt closed | FVCOM ready |
|---|---|---:|---:|---:|---:|---:|---:|
| San Francisco Bay | subgrid boundary spike/sliver | 1 → 0 | 0.875899 → 0.879739 | 44,669 / 84,300 | yes | yes | no |
| Delaware Bay | subgrid boundary spike/sliver | 3 → 0 | 0.852426 → 0.852862 | 169,232 / 302,653 | yes | yes | no |

Both accepted boundaries retained one wet component, the exact open-boundary
geometry, maximum valence 8, zero nonmanifold edges, positive-down depths, and
exact 2DM serialization. The hash-strengthened replay reproduced the tested
boundary bytes exactly.

These are closure proofs rather than full-readiness passes. San Francisco
retains area-transition, slope, first-ring, angle-tail, forcing, and 44
singly-connected-element failures. Delaware retains area-transition, slope,
boundary/field and first-ring, angle-tail, forcing, and 330
singly-connected-element failures. The earlier Delaware deletion yielding two
wet components and 136 singly connected elements remains a mandatory rejection
fixture.
