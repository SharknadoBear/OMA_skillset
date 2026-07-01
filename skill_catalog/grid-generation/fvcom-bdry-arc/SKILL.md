---
name: fvcom-bdry-arc
description: Create QA-ready FVCOM boundary-arc packages from fvcom-region-bpoly RegionBPoly outputs and GSHHG/GSHHS coastline polygons. Use when Codex needs to convert a regional polygon, offshore-side artifact, and robust shoreline topology into classified wet-domain, land-boundary, island-hole, anchor-point, and smooth offshore open-boundary arc artifacts before fvcom-grid-generation.
---

# fvcom-bdry-arc

Use this skill as the second OMA gridding step after `fvcom-region-bpoly` and before `fvcom-grid-generation`.

`fvcom-region-bpoly` chooses the broad four-sided modeling envelope and offshore-side intent. `fvcom-bdry-arc` turns that intent plus GSHHS/GSHHG coastline topology into an explicit boundary package. Mesh generation and SMS `.2dm` writing remain downstream in `fvcom-grid-generation`.

## Core Rule

The bpoly offshore point is a side selector and anchor-search seed. It is not a final boundary endpoint.

Use GSHHS/GSHHG land polygons as the default topology base. CUSP is not the default boundary topology source; use it only through the explicit legacy/debug path when Bear asks for CUSP linework testing.

## Primary Workflow

Run `scripts/run_bdry_arc.py` by default:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --fetch-coastline --run-dir runs/case --name case --mode test --target-resolution-m 250
```

Use an existing GSHHS GeoPackage when already fetched:

```powershell
python scripts/run_bdry_arc.py --region-bpoly-json region_bpoly.json --offshore-artifacts-json offshore_boundary_artifacts.json --coastline-gpkg case_gshhs_land.gpkg --coastline-source gshhs --topology-mode gshhs-vector --run-dir runs/case --name case --mode test
```

Important options:

- `--mode execute|test`, default `execute`. Test mode retains `intermediate/visual_review/`.
- `--coastline-source gshhs|generic-gpkg|cusp-legacy`, default `gshhs`.
- `--fetch-coastline` calls `gshhs-coastline` unless `--coastline-source cusp-legacy` is set.
- `--gshhs-resolution auto|c|l|i|h|f`, default `f`; use `h` later when speed is preferred and topology quality is equivalent.
- `--gshhs-levels`, default `1`.
- `--topology-mode gshhs-vector|iterative-raster|vector-only`, default `gshhs-vector`.
- `--target-resolution-m`, default `250`.
- `--coastline-buffer-km`, default `10`.
- `--seed-mode auto|manual-json`, default `auto`.

## Outputs

Final outputs:

- `bdry_arc_manifest.json`
- `bdry_arc_package.gpkg`
- `bdry_arc_segments.geojson`
- `bdry_arc_review_map.png`

Test-mode visual outputs:

- `intermediate/visual_review/preliminary_arc_map.png`
- `intermediate/visual_review/gshhs_polygon_topology_map.png` for GSHHS-vector mode
- `intermediate/visual_review/gshhs_anchor_arc_map.png` for GSHHS-vector mode
- `intermediate/visual_review/arc_candidate_contact_sheet.png`

Legacy iterative-raster mode may also write `raster_connectivity_iter_XX.png` and `component_classification_iter_XX.png`.

GeoPackage layers:

- `wet_domain`
- `open_boundary_arc`
- `land_boundary_arcs`
- `island_holes`
- `anchor_points`
- `candidate_arcs`
- `coastline_raw`
- `coastline_repaired`
- `topology_diagnostics`
- `forbidden_regions`

## QA Behavior

Mark `final_status: pass` only when the selected arc has valid anchors, avoids extra coastline/land intersections, and creates a seed-containing wet-domain polygon. A recorded `bpoly minus land_union` fallback can pass only when seed, anchor, and open-arc QA remain clean.

Mark `final_status: needs_review` rather than forcing a false pass when:

- GSHHS land polygons or coastline lines are missing;
- the GSHHS vector path falls back to `bpoly minus land_union` and anchors, seed, or open arc QA are not otherwise clean;
- the offshore arc crosses GSHHS land away from its endpoints;
- polygonization cannot close a seed-containing wet-domain face;
- anchors are implausibly far from the intended offshore side;
- the seed is not inside the accepted wet domain.

## Topology Method

The default workflow is GSHHS-first:

1. Load `region_bpoly.json` and `offshore_boundary_artifacts.json`.
2. Fetch or load GSHHS `land_polygons` and `coastline_lines`.
3. Project bpoly, coastline, anchors, arcs, and land polygons to a local UTM CRS.
4. Select coastline anchors nearest the selected bpoly offshore-side endpoints.
5. Generate Bezier and bowed offshore arc candidates using the bpoly offshore azimuth.
6. Polygonize GSHHS coastline lines plus the selected offshore arc and bpoly frame.
7. Choose the seed-containing wet face and subtract GSHHS land polygons.
8. If vector polygonization cannot produce a clean seed-containing face, fall back to `bpoly minus land_union`, record the fallback, and pass only if seed, anchor, and open-arc QA remain clean.
9. Write classified boundary layers and visual-review artifacts.

Use `--topology-mode iterative-raster --coastline-source cusp-legacy` only for legacy CUSP/debug cases. Do not make CUSP the normal topology source until a later refinement workflow explicitly controls it as a local-detail overlay.

Read `references/oceanmesh2d_rpw2019_notes.md` before changing shoreline classification, island filtering, or seeded wet-domain extraction. Treat the local OceanMesh2D snapshot as a method reference only; do not copy GPL MATLAB code into this skill implementation without explicit licensing review.

## Validation

From the skill folder:

```powershell
python scripts/selftest_bdry_arc.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
