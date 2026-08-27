# RegionBPoly standardized output contract

Use this contract when emitting or consuming a RegionBPoly run directory. The
contract standardizes packaging and provenance inside W12; it does not add a
workflow gate or alter coastal review decisions.

## Canonical files

Every run that reaches packaging writes these root-level files:

- `region_bpoly.json`: authoritative RegionBPoly state and QA record;
- `target_region_features.json`: authoritative feature plan used to fit and
  score the polygon;
- `region_bpoly_final_map.png`: whole-domain delivery or review map;
- `offshore_boundary_artifacts.json`: selected-side orientation artifact;
- `region_bpoly_manifest.json`: package state, readiness, file sizes, and
  SHA-256 hashes.

A passing coastal delivery also requires
`region_bpoly_land_side_review.json` and
`region_bpoly_land_side_review.png`. A named-place discovery run additionally
retains `region_place_discovery.json`. Test mode may retain extra intermediate
evidence, but those files are not canonical delivery members.

The manifest excludes its own hash to avoid recursive hashing. Compatibility
aliases such as `<name>_region_bpoly.json` may remain, but consumers must use
the canonical filenames above.

## Feature-plan provenance

Keep schema version `target_region_features_v1` and add the following
backward-compatible fields at document level and on every feature:

- `source_kind`: `explicit`, `catalog_memory`, `web_discovery`,
  `agent_supplied_bbox`, or `unresolved`;
- `source_key`: catalog key, discovery query, or
  `request.target_region_features`;
- `geometry_status`: `user_supplied`, `heuristic_seed`, `discovered_seed`,
  `inferred_seed`, or `unresolved`;
- `purpose`: the feature's operational role.

Catalog and discovered boxes are initial heuristic support, not authoritative
geographic truth. Required-feature coverage and the existing visual gates
remain responsible for acceptance.

## Readiness semantics

`output_package.package_state` and the manifest distinguish:

- `internal_review`: canonical review-state files exist, but the object is not
  deliverable;
- `accepted_delivery`: `final_status` is `pass` and every file required for
  that domain/state exists.

`package_complete` reports file completeness for the current state.
`delivery_ready` is true only for a complete accepted delivery. These are W12
packaging results, not a new RegionBPoly decision gate.
