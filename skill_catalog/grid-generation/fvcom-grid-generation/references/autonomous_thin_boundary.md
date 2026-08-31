# Autonomous Thin-Boundary Closure

Use this reference when `minimal-topology-v1` leaves a superthin component or
when testing the opt-in `autonomous-thin-v1` profile. The profile is an
agent-orchestrated boundary/remesh workflow; it does not change the resolution
of `--conditioning-profile auto`.

## Evidence and decision contract

Use `run_autonomous_thin_workflow.py` as the resumable entry point. Its first
invocation runs `minimal-topology-v1`, writes the diagnostic atlas, and returns
`agent_decision_required` when debt remains. Inspect the emitted images, fill
one pending decision, then invoke the same command with `--decision`; add the
request-bounded `--cusp-gpkg` and regenerate the diagnostic before completing a
shoreline-correction decision. Use `--execute` only after all hashes and source
contracts are present.

1. Run `diagnose_autonomous_thin.py` after minimal conditioning. Bind the mesh,
   boundary-node package, canonical size field, bathymetry, RegionBPoly, case
   manifest, adaptive-resolution manifest, source boundary metadata, boundary
   contract, GSHHS, and any CUSP extract by SHA-256.
2. Inspect the complete-domain map and every component decision diagram.
3. Complete one `fvcom_agent_thin_decision_v1` per component. Set
   `decision_actor.kind=codex_agent`, bind every inspected map by SHA-256, and
   choose exactly one route:
   - `interior_topology_defect`;
   - `resolved_channel_meshing_defect`;
   - `subgrid_boundary_spike_or_sliver`;
   - `subgrid_wet_connection`;
   - `protected_or_source_conflict`.
4. Never infer the decision from JSON alone. Do not add a human-review gate.
5. Regenerate the atlas after every accepted transaction because component
   identifiers and lineage can change.

## Scale and source policy

- Require three elements across an ordinary wet connection. Treat a requested
  resolution as infeasible when width/3 lies below the bathymetry-supported
  floor or its node estimate exceeds the 4,500,000-node planning threshold.
- Derive the local CUSP window buffer as
  `clip(max(10*h, 2*component_diameter, 1 km), 1 km, 5 km)`.
- Keep GSHHS as the closed topology scaffold. Use CUSP only as request-bounded
  evidence or a replacement arc between stable brackets.
- Rank CUSP arcs by bracket coverage, endpoint distance, tangent agreement,
  reported horizontal accuracy, source date, and geometric validity. Reject a
  regularized candidate when either bracket junction turns more than 135°.
  Simplify at
  `max(HOR_ACC, 0.25*h)` and resample at model scale.
- Demote only automatically generated `sharp_turn` or `spit_tip` anchors inside
  the accepted patch. Preserve OBC landfalls, mission features, forcing or river
  anchors, and declared protected geometry.

## Transaction rules

- Repair an interior defect with the existing protected-edge-safe local tools.
- Preserve a resolvable channel and reduce the boundary/field target enough to
  support three elements across before complete remeshing.
- Replace an acute spike/sliver with the best eligible regularized CUSP arc or,
  when none spans stable brackets, a bounded model-scale regularization of the
  existing GSHHS-derived arc.
- Close a subgrid wet connection only as a complete boundary transaction. If
  closure separates an unprotected lobe that is itself unresolved at the
  three-elements-across scale, remove that lobe from the modeled wet domain.
  Never retain multiple wet components.
- Never delete one triangle or a partial incident star. Boundary edits require
  a new adaptive package, canonical bundle, Gmsh-6 mesh, depth sample, and 2DM
  roundtrip.

Try no more than three candidates per component and three remesh cycles. Run a
polygon/OBC/anchor transaction audit before each expensive remesh, retain every
rejected candidate, and accept
only strict superthin improvement with a valid polygon, one manifold wet
component, zero singly connected or nonpositive elements, recovered
constraints, unchanged OBC count/order/forcing lineage, valid positive-down
depths, and exact serialization. Report `autonomous_thin_closed`,
`minimal_local_debt_closed`, and `fvcom_ready` independently.

The historical Delaware passage deletion that produced two wet components and
136 singly connected elements is a mandatory rejection fixture even though it
reached zero superthin triangles.
