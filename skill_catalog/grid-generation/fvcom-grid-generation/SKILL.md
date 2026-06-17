---
name: fvcom-grid-generation
description: Generate, inspect, and quality-check FVCOM-ready unstructured triangular SMS 2DM meshes from local bathymetry and coastline data. Use when Codex needs to build or refine FVCOM grid-generation workflows, create coastline-aware domains with smooth offshore open boundaries, write or parse .2dm files, apply RPWCW2019 size fields and gradation limiting, enforce FVCOM/SMS mesh-quality checks, or plan/test bathymetry-to-grid preprocessing.
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
6. For production-like coastal grids, prefer the coastline-aware workflow:
   - ask the user for the finest target resolution before domain preparation;
   - if the user does not specify it, use the finest bathymetry-grid spacing and record that fallback;
   - prepare the CUSP/CUDEM domain first and stop for visual review;
   - use `--open-boundary-mode auto` by default for non-reference domains so the skill scores bbox-bow, ellipse, and Bezier open-boundary candidates;
  - use `--open-boundary-mode anchor-iterate` when Bear or Codex provides an ocean-direction seed and two rough anchor probes; the optimizer must use the full unpruned coastline for anchor intersections, use that direction as the Bezier tangent from each anchor into the offshore arc, move probes with small target-resolution-scaled steps, freeze an endpoint once it has a valid outer-envelope coastline contact, continue moving only detached endpoints, and snap final accepted anchors to outer-coastline geometry rather than accepting rough seed coordinates;
   - if Codex performs this visual stop, record the reviewer as an agent/AI figure inspection, not a human review;
   - run Gmsh constrained meshing only after `domain_visual_review.json` is marked `pass`.
7. Keep the ellipse workflow only for synthetic tests, quick smoke tests, and debugging.
8. Write the mesh as SMS `.2dm` with `MESH2D`, `MESHNAME`, `E3T`, `ND`, and `NS` records. Keep FVCOM depths positive down.
9. Run quality checks before accepting a mesh:
   - minimum interior angle at least 30 degrees;
   - maximum interior angle at most 130 degrees;
   - maximum bathymetric slope no more than 0.1 where computable;
   - adjacent element area-change metric no more than 0.5;
   - no node connected to more than 8 elements;
   - open-boundary-adjacent triangle interior direction approximately normal to the open boundary.
10. Produce a diagnostic map showing bathymetry, mesh, open-boundary nodes, and failed quality regions. Use this map to tune boundary placement, spacing, and size-field parameters.

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

Prepare a coastline-aware domain and stop for review:

```powershell
python scripts\prepare_coastline_domain.py bathy.nc cusp_coastline.gpkg --run-dir Workspace\Preprocessing\fvcom-grid-generation\runs\case --name case --bbox W S E N --target-resolution-m 100 --open-boundary-mode auto
python scripts\record_domain_review.py --manifest Workspace\Preprocessing\fvcom-grid-generation\runs\case\case_domain_visual_review.json --decision pass --notes "Domain boundary and open boundary visually checked."
```

For the anchor-snapping workflow, first create or inspect the seed map, then run with the rough map direction and two rough probe points:

```powershell
python scripts\prepare_coastline_domain.py bathy.nc cusp_coastline.gpkg --run-dir Workspace\Preprocessing\fvcom-grid-generation\runs\case --name case --bbox W S E N --target-resolution-m 100 --open-boundary-mode anchor-iterate --ocean-direction DX DY --anchor-seeds LON1 LAT1 LON2 LAT2 --anchor-step-factor 1.0 --anchor-min-step-factor 0.1
```

This writes `*_anchor_seed_map.png`, `*_anchor_iteration_review.png`, `*_anchor_points.geojson`, and `*_anchor_report.json`. If the optimizer converges but agent visual inspection shows the arc is too close to shore, too far offshore, or disconnected from the intended model domain, record `needs_followup` and do not mesh.

For validation against a known human mesh, `--reference-2dm Resources\Base_C_D_2m_v2_degree_smooth.2dm` may be supplied. In that mode the reference exterior and first `NS` are reconstructed from the `.2dm`; do not treat that as an independent non-reference open-boundary design.

Generate with Gmsh after review:

```powershell
python scripts\generate_coastline_mesh.py Workspace\Preprocessing\fvcom-grid-generation\runs\case\case_domain_metadata.json bathy.nc --output-2dm Workspace\Preprocessing\fvcom-grid-generation\runs\case\case.2dm --quality-json Workspace\Preprocessing\fvcom-grid-generation\runs\case\case_quality.json
```

Run lightweight workflow self-tests:

```powershell
python scripts\selftest_coastline_workflow.py
```

When running from outside the skill folder, add the skill `scripts/` directory to `PYTHONPATH` or run the scripts from a copied prototype folder that contains `fvcom_grid_generation/`.

## Implementation Notes

- `bathymetry.py` loads local NetCDF or GeoTIFF bathymetry and normalizes to positive-down FVCOM depth.
- `domain.py` infers the deepest bbox side and creates a smooth elliptical offshore arc; the open-boundary nodes are stored separately from the rest of the closed boundary.
- `size_field.py` implements RPWCW-style depth caps, topographic-length-scale slope refinement, and gradation limiting.
- `coastline_domain.py` prepares CUSP/CUDEM coastline-aware domains, applies resolution-based island/thin-water filtering, writes review manifests, and enforces the hard visual gate.
- `open_boundary_designer.py` proposes and scores non-reference open-boundary candidates from bbox-bow, ellipse, and Bezier families. It also implements `anchor-iterate`, which starts from an agent or human visual seed, iteratively intersects a smooth ocean-side Bezier arc with full unpruned coastline linework, honors the ocean-direction seed as the tangent from each anchor into the offshore arc, scores bbox-side touch distance, classifies start/end/middle intersections, freezes endpoints with valid ocean-facing outer-envelope contacts, moves only unresolved endpoints, and snaps the final two accepted anchors to exact coastline points. Agent vision reviews the products; it is not the primary geometry generator.
- `gmsh_builder.py` implements the Gmsh constrained-triangulation path. If `gmsh` is absent, it fails clearly and does not silently fall back to SciPy.
- `mesh_builder.py` uses a deterministic SciPy/Shapely backend for the ellipse workflow.
- `mesh_quality.py` implements the FVCOM manual Chapter 20 quality checks and open-boundary normality diagnostic.
- `sms_2dm.py` handles `.2dm` read/write and SMS-style `NS` nodestrings.

## Gradation

- Default conservative gradation is `g = 0.15`.
- Experimental `g = 0.35` requires explicit user intent and a documented quality review.
- The limiter uses a priority-queue lower-envelope propagation over the structured size grid and enforces `|h_i - h_j| / distance <= g`.
- The limiter only reduces coarser neighboring cells; it must not coarsen cells already made fine by shoreline, feature-size, slope, channel, or depth-cap constraints.
- Record raw size fields, limited size fields, gradation diagnostics, and realized triangle gradation.

## Acceptance

Accept a generated mesh only when:

- the `.2dm` parses successfully after writing;
- all wet nodes have finite positive depth;
- triangles are counterclockwise with positive projected area;
- an explicit open-boundary `NS` string exists;
- the offshore boundary is smooth/curved, ordered, and visually approved before meshing in coastline-aware runs;
- non-reference open-boundary metadata has `design_status="pass_candidate"` or the visual-review notes explicitly explain why meshing is still allowed;
- visual-review manifests distinguish human scientific review from Codex/agent visual inspection;
- no generated triangle falls on land or unresolved filtered islands;
- quality failures are either zero or clearly documented for manual review.
