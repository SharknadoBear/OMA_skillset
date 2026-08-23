# Standard FVCOM Grid Project Contract

Use `scripts/manage_fvcom_grid_project.py` for every new complete grid. The
project is portable: manifests contain only project-relative paths, selected
artifacts are regular files rather than symlinks, and every promotion and
publication verifies SHA-256 before atomic replacement.

```text
<project>/
  project_manifest.json
  project_status.json
  commands.jsonl
  00_request/
  01_region/region_bpoly.json
  02_coastline/coastline.gpkg
  03_boundary/bdry_arc_manifest.json
  04_bathymetry/bathymetry.nc
  05_mesh_intent/case_manifest.json
  05_mesh_intent/size_field.nc
  06_raw_mesh/raw_mesh.2dm
  06_raw_mesh/raw_mesh_manifest.json
  07_conditioning/conditioned_mesh.2dm
  08_audit/final_audit.json
  final/
  logs/
```

Each numbered stage has `_work/` for immutable attempts. `promote` accepts a
source only from that stage's `_work/` and refuses to overwrite a different
selection. `project_manifest.json` uses schema `fvcom_grid_project_v1`.

Every new project records deterministic Gmsh Frontal-Delaunay algorithm 6 as
its operational mesher policy: first-order triangles, one thread, random seed
1, eight native smoothing steps, and no fallback. Generate the raw candidate
with `run_mesher_portfolio_case.py`; omitting `--candidates` executes Gmsh-6
only. Promoting `raw_mesh.2dm` requires its project-local
`candidate_manifest.json`. The manager validates the candidate and generator
report, then writes the portable `fvcom_raw_mesh_provenance_v1` sidecar above.
Clean-room and Gmsh 1/5 candidates remain explicit research controls and cannot
be promoted or published through this operational project contract.

When a raw or conditioned terminal mesh exists, `publish` requires the mesh
quality, conditioning, boundary-node, OBC-remap, roundtrip, and review-map
companions. It writes `final/fvcom_grid.2dm` even for a non-ready terminal mesh,
plus `final/fvcom_grid_status.json`; the status uses schema
`fvcom_grid_delivery_v1` and records readiness, submission eligibility, OBC and
forcing status, selected-stage hashes, and failure taxonomy. A pre-mesh failure
writes the status but never fabricates a 2DM.

Run `validate --require-submission-ready` immediately before future job
submission. It requires `submission_eligible=true`, `fvcom_ready=true`, and an
exact final-mesh hash. A stable filename alone is never sufficient.
