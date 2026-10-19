# State and worker manifest contract

`project_status.json` uses `simulation_project_status_v1` with `case_id`, `state`, `updated_at`, `active_grid_case`, `active_attempt`, `workflow_status`, `scientific_assessment`, `blocking_reasons`, and an append-only `history` array. A transition appends the prior state and evidence; it never erases prior attempts.

Each of the three preparation roles returns this common envelope, including a role performed directly by the simulation lead when agent slots are limited. A role may use a more specific
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

For new barotropic projects, store each preparation handoff as `role_manifest.json` under its fixed role root from the [barotropic workspace contract](barotropic_agent_layout.md). The manifest's `artifact_root` is that project-relative role path. `project_manifest.json` records `agent_layout.schema=fvcom_barotropic_agent_layout_v1` with the `lead`, `grid`, `tpxo`, and `configuration_build` entries in `agent_layout.role_roots`. Existing projects without this field retain their recorded legacy paths; initialization does not rewrite an existing project manifest to adopt the new layout.

## Resume rules

- Verify all hashes before reusing a completed stage.
- If a hash changed, do not overwrite or silently resume; create a new attempt or re-enter the earliest invalidated state.
- Only one stability job may be active.
- The executable hash is frozen before the first submitted attempt and must be identical across the accepted/fresh pair.
- Accepted and fresh grid artifacts have independent roots and forcing products even when deterministic generation yields identical bytes.

## paper_barotropic exception

The fresh-only paper profile has independent smoke and production attempt IDs
`attempt_smoke_0000..0002` and `attempt_production_0000..0002`. Counts persist
across resumes and corrections. There is no accepted/fresh pair, exhaustive
stability strategy or benchmark. `attempt_limit_reached` means a stage failed
all three total attempts; it is a failure, not validation completion. An earlier
explicit inability may use `scope_exhausted` with its evidence without exhausting
the ordinary strategy. Bear owns timing and acceptance; these native manifests
are evidence records, not an autonomous campaign controller.
