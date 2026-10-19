# Retained accepted-grid forcing join

Use only for a previously accepted mesh with unchanged coordinates, depths and
connectivity after intentional preprocessing cleanup. Fresh models continue
through the full standardized project gates. The supported retention record is
`fvcom_grid_post_cleanup_integrity_v1`, with passing final-mesh audit and all
protected t6v6 files matching; other retention formats require an explicit owning
extension rather than relabeling a failed or unknown receipt.

The request has `schema_version=fvcom_retained_grid_join_request_v1`,
`workflow_kind=accepted_retained`, and an `inputs` object. Each input is
`{path, sha256}`; relative paths resolve from the request directory:

- `grid_contract`: delivered `fvcom_grid_delivery_contract_v1`, binding the current
  mesh and boundary, quality, remap and source-bound TGE companions.
- `original_mesh`, `original_boundary_nodes`, `original_publication_status`:
  preserved original bytes, also present in the retained source snapshot.
- `source_case_manifest`, `open_exterior_source`, `boundary_resolution_source`:
  exact retained case and selected strict/Adaptive boundary authorities.
- `tge_source`: exact source used by the frozen executable/preconfiguration join.
- `retention_evidence`: passing post-cleanup integrity record.
- `source_snapshot_manifest`: `fvcom_retained_source_snapshot_v1`; its `files`
  each bind `retained` paths and hashes relative to the manifest's grandparent
  (the gridding-role root). Every retained member is rehashed.
- `obc_points`, `forcing`, `forcing_manifest`: terminal ordered-node coordinates,
  actual NetCDF elevation and the ready TPXO owner manifest.
- `preconfiguration_manifest`: the ready DAT package with generated-file hashes.
  Historical relative generated-file paths may relocate beside their manifest
  using their basename, only if exact hashes and content checks pass.

An optional `supplementary_retention_evidence` binding preserves the cleanup log.
Only the immutable certificate directory is written after all gates pass. It
contains the certificate, exact auditor source, and verification receipt. Replay
requires the recorded auditor version and rehashes/re-audits every dependency.
Original publication status and source-order remaps are not edited. The lead
binds this typed certificate in preparation and explicitly records the retained
route instead of claiming standardized-project success.

Stable OBC IDs must match the selected Adaptive source IDs. A case display alias
may differ only when the exact historical case explicitly selects that source
ID and the terminal nodestring, ordered nodes and cyclicity agree. Cyclic chains
cannot be trimmed. Noncyclic trims are restricted to endpoint(s), must reproduce
the original sum-five TGE failure, and must pass the complete corrected audit.

An explicitly selected `--endpoint-fan-max-steps N` correction may additionally
return a multi-node contiguous endpoint trim. The terminal contract must bind
`source_delivery.endpoint_fan_max_steps` and the complete hash-bound
`delivery_artifacts.endpoint_fan_trace`. The trace names the source mesh and
TGE hashes, planner hash, all original chains and cyclicity, the uniquely
anchored original fatal components, every current-endpoint demotion, and each
before/after source-audit and ordered-chain digest. The join recomputes the
planner from the original mesh and exact source, compares the entire trace,
and requires exact terminal geometry and plural order. Missing, stale, edited,
reordered, enlarged-bound, unanchored or incomplete traces fail closed; presence
of a trace alone never waives the ordinary quality/boundary/DAT/forcing gates.

New certificates archive both `auditor.py` and `fvcom_tge_endpoint_fan.py` and
bind both hashes. Verification requires the executing planner to match the
archived planner before scientific replay. Run with the matching frozen Grid
package on `PYTHONPATH`; the archived helper remains beside the auditor. Earlier
certificates retain their earlier exact auditor versions and are not rewritten.

Run `selftest_retained_grid_join.py`, `selftest_forcing_join.py` and
`selftest_grid_project.py` after changes. Also retain the actual failing legacy
join, a passing real retained-grid replay, and independent mutations of its
bindings, geometry, boundary/identity, TGE, DAT and forcing evidence. Generic
identity fixtures alone do not certify a regional mesh.
