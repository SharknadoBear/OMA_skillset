---
name: fvcom-grid-generation
description: Generate FVCOM-ready SMS 2DM meshes from fvcom-bdry-arc boundary-loop packages and CUDEM/NBS/CRM/ETOPO fallback bathymetry using a clean-room Python OceanMesh/RPW-style constrained Delaunay refinement workflow. Use when Codex needs gridding after fvcom-region-bpoly and fvcom-bdry-arc, boundary node-string assignment, bathymetry-source provenance, progress-heartbeat artifacts, FVCOM mesh QA, or Delaware-style staged grid smoke tests.
---

# fvcom-grid-generation

Use this skill as the third OMA gridding step:

```text
fvcom-region-bpoly -> fvcom-bdry-arc -> cudem-bathy -> fvcom-grid-generation
```

`fvcom-bdry-arc` owns boundary-arc generation. This skill consumes its continuous model-boundary-loop product, fetches or reuses CUDEM-first fallback bathymetry, builds a mesh-size field, generates a pure-Python OceanMesh-style triangular mesh, writes FVCOM/SMS `.2dm`, and runs QA.

## Core Rules

- Do not redesign the bpoly or boundary arc inside this skill. Reuse `region_bpoly.json`, `offshore_boundary_artifacts.json`, `bdry_arc_manifest.json`, and `model_boundary_loops.gpkg`.
- Use `cudem-bathy` for bathymetry fetches. Generated runs should call `fetch_bathy_sources.py` with CUDEM/NBS/CRM/ETOPO fallback unless the user explicitly supplies a bathymetry NetCDF or chooses a narrower policy.
- Treat OceanMesh2D as a method reference only. Do not copy or translate GPL-3.0 MATLAB source line-by-line.
- Keep node depths positive down in FVCOM outputs.

## Primary CLI

Run `scripts/run_fvcom_grid.py`.

Use existing upstream artifacts:

```powershell
python scripts/run_fvcom_grid.py --bdry-arc-manifest bdry_arc_manifest.json --boundary-loops-gpkg model_boundary_loops.gpkg --bathy-nc bathy.nc --run-dir Workspace/Preprocessing/fvcom-grid-generation/runs/case --name case --mode test
```

Run the full chain from a prompt:

```powershell
python scripts/run_fvcom_grid.py --request-text "Delaware River estuary FVCOM grid" --run-dir Workspace/Preprocessing/fvcom-grid-generation/runs/delaware --name delaware --mode test --coarse-smoke
```

Important options:

- `--request-text`: run upstream bpoly, boundary-arc, and CUDEM steps when artifacts are not supplied.
- `--region-bpoly-json`, `--offshore-artifacts-json`, `--bdry-arc-manifest`, `--boundary-loops-gpkg`, `--bathy-nc`: reuse existing artifacts.
- `--coarse-smoke`: set land/island boundary spacing to `250 m` and open-boundary spacing to `5000 m`.
- `--land-spacing-m`, default `50`.
- `--open-spacing-m`, default `3000`.
- `--gradation`, default `0.15`.
- `--target-timestep-s auto|SECONDS`, default `auto`.
- `--max-interior-points`, default `80000`.
- `--bathy-fallback-policy`, default `cudem-nbs-crm-etopo`.
- `--bathy-resolution-policy`, default `source-priority`.
- `--bathy-target-spacing-arcsec`, default `1.0`.
- `--bathy-max-sources`, default `256`.
- `--progress-interval-s`, default `10`.
- `--size-field-max-cells`, default `1500000`; full-resolution bathymetry is still used for node-depth sampling, but size-field calculations use a bounded working grid.

## Outputs

Every run writes:

- `fvcom_grid.2dm`
- `fvcom_grid_manifest.json`
- `mesh_quality.json`
- `mesh_review_map.png`
- `size_field.nc`
- `size_field.png`
- `boundary_nodes.geojson`
- `mesh_nodes_elements.gpkg`
- `progress.json`
- `progress.jsonl`

Generated bathymetry runs also write the `cudem-bathy` fallback NetCDF, metadata JSON, source-id map, and health-check JSON under `upstream/cudem_bathy/`.

## Method

Read `references/oceanmesh_rpw2019_method.md` before changing the mesh generator or size field. Read `references/fvcom_sms_quality.md` before changing `.2dm`, `NS`, or QA behavior. Read `references/cudem_dependency.md` before changing bathymetry fetch wiring.

The v1 clean-room backend:

1. Read the accepted wet-domain polygon and classified boundary segments from `model_boundary_loops.gpkg`.
2. Densify boundary arcs by segment class: coarse on open boundary, fine near land/islands.
3. Build a RPW/RPWCW-style size field from shoreline distance, bathymetry depth caps, topographic length scale, optional CFL cap, and gradation limiting.
4. Generate a SciPy Delaunay mesh from fixed boundary nodes plus interior candidate nodes.
5. Recover boundary constraints by iterative midpoint insertion until fixed boundary segments are present or `boundary_constraint_not_recovered` is reported.
6. Refine over-large or poor triangles with circumcenter/edge midpoint insertion, then smooth only interior nodes.
7. Interpolate fallback bathymetry depths to nodes, write `.2dm`, and run FVCOM quality checks.

For long runs, inspect `progress.json` for the latest stage, percent, elapsed time, subprocess PID, and last known artifact. Do not lower bathymetry resolution or abort a slow case only because no terminal output is visible.

When fallback bathymetry is much finer than the mesh-size calculation needs, keep the fetched bathymetry unchanged and downsample only the in-memory size-field working grid. Record the source cell count and size-field cell count in `fvcom_grid_manifest.json`.

## Acceptance

Accept a mesh only when:

- `.2dm` roundtrips;
- all wet-node depths are finite and positive;
- triangles are counterclockwise with positive area;
- the open-boundary `NS` nodestring is present and ordered unless the upstream domain explicitly has no ocean open boundary;
- min angle is at least `30 deg`;
- max angle is at most `130 deg`;
- max bathymetric slope is no more than `0.1` where computable;
- adjacent element area-change metric is no more than `0.5`;
- node valence is no more than `8`;
- boundary constraints are recovered.

If strict gates fail, still write artifacts and mark `final_status: needs_review` with failure taxonomy.

## Validation

From the skill folder:

```powershell
python scripts/selftest_fvcom_grid.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
