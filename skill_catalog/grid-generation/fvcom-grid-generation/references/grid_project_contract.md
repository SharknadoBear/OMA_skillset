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
  08_audit/mesh_review_map.png
  08_audit/mesh_review_map_manifest.json
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

For a coastal project, the promoted boundary must carry a passing
`fvcom_coastline_source_coverage_v1` contract. It proves that the RegionBPoly
was centered inside at least a 2x GSHHS source footprint, all landfalls came
from physical coastline lines, the delivered exterior has zero source-frame
dependency, and the whole/zoom coverage maps and hashes remain current.
Historical exact-bbox packages are diagnostic-only for new projects.

The boundary must also carry an active passing `fvcom_open_exterior_contract_v2`
when residual roles are enabled (or v1 for the strict-reject legacy policy).
Every non-OBC residual must have an accepted role with current whole/component
map bindings. Missing, stale, pending, report-only, unassigned, or unsupported
v3 evidence blocks mesh readiness and publication. RegionBPoly truncation is
resolved upstream by `fvcom-region-bpoly`, not by this project contract.

When a raw or conditioned terminal mesh exists, `publish` requires the mesh
quality, conditioning, boundary-node, OBC-remap, and roundtrip companions. It
automatically writes the standard positive-down bathymetry/triangle review map,
with the delivered OBC in red, a bounded Esri topographic background or
project-local coastline fallback, and the grid name plus four-decimal
`q_L3sigma` in the title. The PNG and hash-bound manifest are stable under
`08_audit/` and copied to `final/`. Publication writes `final/fvcom_grid.2dm`
even for a non-ready terminal mesh, plus `final/fvcom_grid_status.json`; the
status uses schema `fvcom_grid_delivery_v1` and records the bound quality-policy
hash, benchmark readiness, regional debt, submission eligibility, OBC/forcing
status, selected-stage hashes, and Class-1 failure taxonomy. A pre-mesh failure
writes the status but never fabricates a 2DM or map.

Run `validate --require-benchmark-ready` before a first benchmark run. Run
`validate --require-submission-ready` immediately before future job submission;
it additionally requires forcing/remap compatibility, complete provenance, and
an exact final-mesh hash. Class-2/3 debt does not independently block either
decision. A stable filename alone is never sufficient.
