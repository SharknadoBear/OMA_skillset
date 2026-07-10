# FVCOM SMS 2DM Quality Reference

Use this note when changing `.2dm`, OBC nodestring, depth, constraint, or acceptance behavior.

## Output Invariants

- Write `MESH2D`, `MESHNAME`, `E3T`, `ND`, and `NS` records.
- Keep node depths finite and positive down.
- Keep triangles counterclockwise with positive projected area.
- Write one ordered `NS` nodestring unless upstream metadata explicitly defines no ocean boundary.
- Require every consecutive OBC node pair to be an actual mesh boundary edge.
- Preserve every constraint selected by the postprocessing boundary policy.

## Default Gates

- minimum triangle angle: `30 deg`;
- maximum triangle angle: `130 deg`;
- maximum bathymetric slope: `0.1`;
- maximum adjacent element area-change metric: `0.5`;
- maximum true vertex-neighbor valence: `8`;
- one manifold, traversable component;
- no missing protected constraints or nonpositive elements.

When any gate fails, retain all artifacts and set `final_status: needs_review`. Normal generation writes one quality document for the generation-time smoothed mesh. Cleanup comparisons belong to the standalone postprocessor.

Adaptive boundary packages additionally require ordered explicit chains, per-node target spacing, and OBC size compatibility: 95th-percentile `L/h <= 1.55` and maximum `L/h <= 2.0`.
