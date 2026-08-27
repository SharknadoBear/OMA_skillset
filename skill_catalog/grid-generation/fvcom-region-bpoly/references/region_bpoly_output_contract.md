# RegionBPoly standardized output contract

Use this contract when emitting or consuming a RegionBPoly run directory. The
contract standardizes packaging and provenance inside W12; it does not turn
coastal review uncertainty into a downstream intake gate.

## Canonical files

Every run that reaches packaging writes these root-level files:

- `region_bpoly.json`: authoritative RegionBPoly state and QA record;
- `target_region_features.json`: authoritative feature plan used to fit and
  score the polygon;
- `region_bpoly_final_map.png`: whole-domain delivery or review map;
- `offshore_boundary_artifacts.json`: selected-side orientation artifact;
- `region_bpoly_manifest.json`: package state, readiness, file sizes, and
  SHA-256 hashes.

A finalized coastal delivery also retains
`region_bpoly_land_side_review.json`. A clean visual pass additionally requires
`region_bpoly_land_side_review.png`; when source maps are stale or unusable,
the missing compact map is recorded as a warning rather than blocking the
usable geometry. A named-place discovery run additionally retains
`region_place_discovery.json`. Test mode may retain extra intermediate evidence,
but those files are not canonical delivery members.

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

- `internal_review`: a nonterminal `review_pending` or `repair_required` state;
- `accepted_delivery`: the latest valid RegionBPoly has been finalized with
  `final_status: pass`, either cleanly or with explicit review warnings.

`package_complete` reports completeness of the usable geometry package for the
current state. `delivery_ready` is true for a complete accepted delivery.
Downstream consumers preserve all review provenance but must not reject usable
RegionBPoly geometry because of review, package-state, or readiness labels.
These are W12 packaging results, not a new RegionBPoly decision gate.
