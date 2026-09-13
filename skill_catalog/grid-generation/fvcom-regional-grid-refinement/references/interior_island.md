# Single interior solid-island transaction

Use `scripts/interior_island.py` only when an explicitly scoped scientific
request calls for representing one previously unrepresented interior land
feature. This is a separate supplied-source sensitivity/repair route. It does
not relax the unchanged-retained join or the GSHHS-only autonomous-thin profile.
A CUSP ring must remain labelled CUSP, including estimated segments, source
year, horizontal transform, and tidal/terrain datum limitations. A terrain-zero
contour is not automatically a shoreline. Generalize dense source geometry to
the declared model scale and preserve the raw source and transformation ledger.

The existing refinement-v2 request remains topology preserving. An island
adapter must call `validate_request(path)` with the separate
`fvcom_single_interior_island_request_v1` document. Its executable validator is
the authoritative schema; every top-level field and every binding is required,
and unknown fields fail. Required fields are:

- `schema_version`, `source_crs`, `metric_crs` (metres), and a nonempty
  `scientific_definition` describing the actual land representation;
- `buffer_candidates_m`: one to three increasing, positive bounded patch buffers;
- `core_target_m`, `gradation` in `(0,0.10]`, `maximum_patch_nodes` in
  `1..1000000`, and `area_tolerance_m2` in `(0,0.1]`;
- `inputs`, containing exactly `source_mesh`, `source_grid_contract`,
  `source_certificate`, `island_ring`, `source_evidence`, `bathymetry_receipt`,
  `protected_contract`, `tge_source`, and `quality_policy`. Each is a regular-file
  `{path, sha256}` binding. The parent retained certificate must have passed and
  bind this mesh, contract and TGE source. This first route does not recursively
  repair a previously repaired mesh.

`island_ring` contains only `{crs, coordinates}`: metric coordinates in open
ring order, with no repeated closure point. The protected contract contains
only `protected_node_ids` (unique positive SMS IDs) and `open_boundaries`, copied
exactly from the bound source grid contract. Include all OBC and river identities.
The source-evidence and bathymetry receipts bind their raw inputs, scripts,
scientific choices, coverage and visual/test reviews. Their scientific validity
is assessed by the caller; a hash does not prove a survey or datum claim.

## Preparation and generation

1. Read the source with `IdentityMesh.read`, which rejects duplicate and
   noncontiguous IDs and preserves E3T materials, coordinates, depths and all
   nonentity records. Project with `IdentityMesh.projected`.
2. For each declared buffer, call `select_patch`. Selection uses triangle/polygon
   intersections, including a feature wholly inside a triangle with no source
   vertex. Close complete stars. The new island must be strictly inside the
   existing wet domain; old physical boundaries may touch the patch but remain
   exact. A patch cannot remove a protected node.
3. Build a wet field on selected retained vertices outside the new land plus
   all obstacle/ring vertices. Use authoritative incumbent targets if available;
   otherwise explicitly record `incumbent_sizes`, the median incident edge
   length proxy. `WetSizeField` computes a lower envelope on a visibility-safe
   finite graph. Continuous ring-distance seeds require wet-visible connecting
   segments; graph edges and callback cones cannot cross land. This is not a
   claim of exact continuous geodesics. The graph has a 5,000-sample bound.
4. Freeze the selected mask, actual patch loops, seam/physical edges, ring,
   field samples/values/edges, source/code versions and request before meshing.
   `generate_patch` independently checks stitch field p95<=1.5/max<=2.0 and
   uses the owning Gmsh4.15.2 algorithm6, one thread, seed1, order1, eight
   smoothing steps, locked source chords, and no fallback.
5. Keep RAW patch MSH, nodes, connectivity and generator provenance. Restrict
   any required `minimal-topology-v1` conditioning to the patch; no source
   boundary edits, outside movement, global retriangulation or smoothing.
6. `merge_patch` reuses vacated node/element IDs deterministically and appends
   surplus IDs. It rejects fewer replacement entities, mixed source materials,
   nonpositive patch areas and invalid new bathymetry. Never fill gaps with
   unused/degenerate nodes or renumber the retained mesh. New depths use a
   caller-supplied, hash-bound strict sampler on serialized-coordinate inputs;
   retained depths and records remain exact. No implicit extrapolation or datum
   correction is available.

## Independent certificates and downstream admission

`audit_transaction` rereads the complete old and new serialized meshes. It
requires exact retained node and outside element records, materials and NS
records; contiguous, unique, used entities; original physical chords; two-sided
stitches; exactly one added simple ring; one wet component; Euler delta -1;
the declared area loss; and no overlapping/gapped/land-covering patch triangles.
It checks actual signed areas without repairing their orientation first.

This structural receipt alone has `submission_eligible: false`.
`audit_quality_and_tge` recomputes the central full-mesh quality policy and
literal source-bound TGE test and verifies an exact serialized round trip.
Do not invent new baseline vetoes for ordinary angle/transition/slope advisories.
Require the explicit local size/shoreline-resolution gates separately.

Call `certify_geometry` with the frozen field NPZ, RAW patch NPZ, numbering
ledger, verified strict sampler, selected declared buffer and review bindings.
It independently reconstructs the selection and wet field, repeats Gmsh,
replays numbering and every new serialized depth, and requires exact complete
mesh bytes. It then repeats structure, complete boundary-sidecar, full quality,
TGE and round-trip gates. The v1 certificate supports unconditioned RAW patches
only; conditioning needs an explicit future replay contract, not an equality
waiver. Evidence must include `source_review`, `physical_review`,
`boundary_review`, `boundary_sidecar`, `sampler_code` and
`independent_transaction_review` regular-file bindings. Relative evidence paths
resolve against the request directory. Scientific review remains the caller's
responsibility; file hashes alone do not establish a physical interpretation.

`audit_boundary_sidecar` preserves every old Point feature dictionary and checks
complete once-only chain edges, coordinates, OBC roles, IDs and cyclicity.
New ring nodes require `source_type: supplied_interior_island`,
`boundary_kind: island`, `is_hard_anchor: true`, no retained source-node ID or
index claim, and hash-bound `source_geometry` and `source_review` matching
the request. Relative paths in those features resolve against the sidecar.
The resulting `fvcom_single_interior_island_geometry_certificate_v1` binds
request, frozen selection, original/RAW/delivered hashes, source/code/policy
hashes, numbering lineage and actual visual review; submission remains false.
Record this composite as a retained mesh plus local Gmsh patch, never as a fully
regenerated RAW candidate. The original retained certificate must reject it.

Avoid a certificate/contract hash cycle: certify geometry first; create a new
`fvcom_grid_delivery_contract_v1` binding that certificate and complete boundary
sidecars; then make a separate handoff receipt binding regenerated
preconfiguration, grid-edge and station mappings and a fresh exact-OBC forcing
audit. Unchanged OBC coordinates permit forcing reuse only after actual-content
verification; an old forcing/retained certificate cannot be copied as admission.
Datum interpretation, forcing compatibility and subsequent model stability
remain distinct assessments. Preserve all failed candidates and the campaign's
bounded repair limits. No code here submits model work.

Run `scripts/selftest_interior_island.py`, existing regional `self_test.py`, and
applicable Grid retained-join, forcing-join, quality, Gmsh and TGE regressions.
Add a bounded actual-case replay and independent mutation tests before release.
