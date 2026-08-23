# Autonomous Thin v1 Three-Estuary Evidence — 2026-08-22

Three clean, sequential Codex agents forward-tested installed commit `aa38349`.
Large artifacts remain in the preprocessing workspace.

| Case | Final nodes / triangles | Route | Superthin | Max valence / >8 | q_L3σ | Thin closed | Minimal closed | FVCOM ready |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Galveston–Trinity Bay | 217,453 / 406,144 | no-op | 0 | 8 / 0 | 0.869360 | yes | yes | no |
| Tampa Bay | 60,353 / 107,107 | subgrid wet connection | 0 | 9 / 10 | 0.848930 | yes | no | no |
| Lower Columbia River estuary | 44,138 / 77,025 | interior topology | 68 / 30 components | 17 / 831 | 0.572217 | no | no | no |

Galveston proves the no-op path. Tampa proves complete subgrid-connection
closure can remove the thin component without splitting the wet domain, while
also proving thin closure is independent of valence and full readiness.
Columbia defines the current capability limit: one accepted local transaction
reduced superthin debt 70→68 and components 31→30, after which the global
conditioner could not make an admissible monotonic improvement. The workflow
stopped after cycle 2 and retained its rollback champion.

All cases passed their fresh domain/arc/bathymetry inputs, positive-depth and
2DM serialization checks. None is authorized for production FVCOM execution.
Recurring readiness debt includes OBC forcing reconciliation and singly
connected elements; Windows deep-path publication also required scientifically
identical short-path reruns in Tampa and Columbia.
