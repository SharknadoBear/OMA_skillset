# State and worker manifest contract

`project_status.json` uses `simulation_project_status_v1` with `case_id`, `state`, `updated_at`, `active_grid_case`, `active_attempt`, `workflow_status`, `scientific_assessment`, `blocking_reasons`, and an append-only `history` array. A transition appends the prior state and evidence; it never erases prior attempts.

Every initial worker returns this common envelope. A role may use a more specific
`schema`/`schema_version` identifier, but all fields below and their semantics are
mandatory:

```json
{
  "schema": "role_specific_worker_manifest_v1",
  "worker": "grid|tpxo|configuration_build",
  "status": "ready|waiting_user|blocked|failed",
  "artifact_root": "project-relative path",
  "hashes": {"logical_name": "sha256"},
  "provenance": {},
  "warnings": [],
  "blocking_reasons": [],
  "resume_token": "stable unique string or structured token"
}
```

`ready` requires all role outputs, not merely a started task. `waiting_user` is limited to secure authentication, registered-data authorization, or an explicit user stop. `blocked` identifies an external or scope boundary. `failed` identifies invalid generated evidence.

## Resume rules

- Verify all hashes before reusing a completed stage.
- If a hash changed, do not overwrite or silently resume; create a new attempt or re-enter the earliest invalidated state.
- Only one stability job may be active.
- The executable hash is frozen before attempt 0001 and must be identical across both Galveston grid cases.
- Accepted and fresh grid artifacts have independent roots and forcing products even when deterministic generation yields identical bytes.
