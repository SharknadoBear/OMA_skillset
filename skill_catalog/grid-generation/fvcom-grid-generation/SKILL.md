---
name: fvcom-grid-generation
description: Generate, inspect, and quality-check FVCOM-ready unstructured triangular SMS 2DM meshes from local bathymetry. Use when Codex needs to build or refine FVCOM grid-generation workflows, create smooth non-square offshore open boundaries, write or parse .2dm files, enforce FVCOM/SMS mesh-quality checks, or plan/test bathymetry-to-grid preprocessing.
---

# FVCOM Grid Generation

## Purpose

Use this skill to turn local bathymetry into an FVCOM-ready SMS `.2dm` mesh with an explicit open-boundary nodestring. Prefer this skill when the task involves grid generation, `.2dm` parsing/writing, offshore boundary design, bathymetry interpolation, mesh-quality checks, or map-based mesh inspection.

## Workflow

1. Work inside `Workspace/Grid_preprocessing/` for project trials, runs, plots, and reports.
2. Use local bathymetry first. Do not fetch public bathymetry unless the user explicitly asks for a fetch/cache workflow.
3. Read `references/fvcom_chapter20_sms_guidance.md` before changing mesh-quality thresholds, open-boundary handling, or `.2dm` boundary strings.
4. Read `references/rpwcw2019_mesh_guidance.md` before changing element-size fields, gradation, shoreline/slope/channel refinement, or bathymetry smoothing assumptions.
5. Generate a smooth offshore boundary. The v1 default is an elliptical offshore arc, tagged as the open boundary, rather than a square bbox edge.
6. Write the mesh as SMS `.2dm` with `MESH2D`, `MESHNAME`, `E3T`, `ND`, and `NS` records. Keep FVCOM depths positive down.
7. Run quality checks before accepting a mesh:
   - minimum interior angle at least 30 degrees;
   - maximum interior angle at most 130 degrees;
   - maximum bathymetric slope no more than 0.1 where computable;
   - adjacent element area-change metric no more than 0.5;
   - no node connected to more than 8 elements;
   - open-boundary-adjacent triangle interior direction approximately normal to the open boundary.
8. Produce a diagnostic map showing bathymetry, mesh, open-boundary nodes, and failed quality regions. Use this map to tune boundary placement, spacing, and size-field parameters.

## Scripts

The Python toolbox lives under `scripts/fvcom_grid_generation/`.

Common commands from the skill folder:

```powershell
python scripts\roundtrip_2dm.py Resources\Base_C_D_2m_v2_degree_smooth.2dm --output-2dm Workspace\Grid_preprocessing\runs\roundtrip.2dm
python scripts\quality_report.py Workspace\Grid_preprocessing\runs\roundtrip.2dm --output-json Workspace\Grid_preprocessing\runs\roundtrip_quality.json
python scripts\generate_synthetic_mesh.py --run-dir Workspace\Grid_preprocessing\runs\synthetic_smoke
python scripts\generate_from_bathymetry.py local_bathy.nc --output-2dm Workspace\Grid_preprocessing\runs\domain_grid.2dm --quality-json Workspace\Grid_preprocessing\runs\domain_quality.json
python scripts\fvcom_grid_generation\view_mesh.py Workspace\Grid_preprocessing\runs\domain_grid.2dm --output-png Workspace\Grid_preprocessing\runs\domain_grid.png
```

When running from outside the skill folder, add the skill `scripts/` directory to `PYTHONPATH` or run the scripts from a copied prototype folder that contains `fvcom_grid_generation/`.

## Implementation Notes

- `bathymetry.py` loads local NetCDF or GeoTIFF bathymetry and normalizes to positive-down FVCOM depth.
- `domain.py` infers the deepest bbox side and creates a smooth elliptical offshore arc; the open-boundary nodes are stored separately from the rest of the closed boundary.
- `size_field.py` implements RPWCW-style depth caps, topographic-length-scale slope refinement, and gradation limiting.
- `mesh_builder.py` uses a deterministic SciPy/Shapely backend for the v1 testable workflow. A future Gmsh backend can share the same input/output interfaces.
- `mesh_quality.py` implements the FVCOM manual Chapter 20 quality checks and open-boundary normality diagnostic.
- `sms_2dm.py` handles `.2dm` read/write and SMS-style `NS` nodestrings.

## Acceptance

Accept a generated mesh only when:

- the `.2dm` parses successfully after writing;
- all wet nodes have finite positive depth;
- triangles are counterclockwise with positive projected area;
- an explicit open-boundary `NS` string exists;
- the offshore boundary is smooth/curved, not a square bbox side;
- quality failures are either zero or clearly documented for manual review.
