---
name: fvcom-bdry-arc
description: Create QA-ready FVCOM boundary-arc and continuous model-boundary-loop packages from fvcom-region-bpoly RegionBPoly outputs and GSHHG/GSHHS coastline polygons. Use when Codex needs to convert a regional polygon, offshore-side artifact, and robust shoreline topology into a gridding-ready model-domain polygon, classified land/model outer boundary, island boundaries, and smooth offshore open-boundary arc artifacts before fvcom-grid-generation.
---

# fvcom-bdry-arc

Use this skill as the second OMA gridding step after `fvcom-region-bpoly` and before `fvcom-grid-generation`.

`fvcom-region-bpoly` chooses the broad four-sided modeling envelope and offshore-side intent. `fvcom-bdry-arc` turns that intent plus GSHHS/GSHHG coastline topology into an explicit boundary package and continuous model-boundary loop package. Mesh generation and SMS `.2dm` writing remain downstream in `fvcom-grid-generation`.

## Core Rule

The bpoly offshore point is a side selector and anchor-search seed. It is not a final boundary endpoint.

The two open-boundary anchors are coastline-on-bpoly points. For the selected offshore bpoly side, find the two adjacent bpoly sides, intersect each adjacent side with the GSHHS coastline/land boundary, and choose the crossing closest to the adjacent offshore corner. If GSHHS does not intersect exactly because of projection or resolution, snap/node only within the target-resolution tolerance; otherwise mark `needs_review`.

The selected offshore side's original two corners are control points, not final anchors. The default GSHHS workflow splits the adjacent bpoly sides at the coastline anchors, builds the seaward chain `start_coast_anchor -> start_offshore_corner -> offshore_side -> end_offshore_corner -> end_coast_anchor`, and deforms that chain into a smooth open boundary.

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

Final outputs from every normal `run_bdry_arc.py` run:

- `bdry_arc_manifest.json`
- `bdry_arc_package.gpkg`
- `bdry_arc_segments.geojson`
- `bdry_arc_review_map.png`
- `model_boundary_loop_manifest.json`
- `model_boundary_loops.gpkg`
- `model_boundary_segments.geojson`
- `model_boundary_colored_map.png`

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
- `frame_clip_boundary_arcs`
- `island_holes`
- `anchor_points`
- `candidate_arcs`
- `coastline_raw`
- `coastline_repaired`
- `topology_diagnostics`
- `forbidden_regions`

`run_bdry_arc.py` automatically calls the model-boundary loop builder after `bdry_arc_package.gpkg` is written. Keep `scripts/build_model_boundary_loops.py` as a standalone debug/rebuild utility when an existing package must be reclassified without rerunning coastline fetch or arc selection.

The loop package writes these layers:

- `model_domain_polygon`
- `model_outer_boundary`
- `model_outer_boundary_segments`
- `island_boundary_polygons`
- `island_boundary_lines`
- `source_open_boundary_arc`

## QA Behavior

Mark `final_status: pass` only when the selected arc keeps the coastline/bpoly anchor endpoints, avoids extra coastline/land intersections, is present on the final wet-domain boundary, and creates a seed-containing wet-domain polygon.

Mark `final_status: needs_review` rather than forcing a false pass when:

- GSHHS land polygons are missing;
- the coastline-on-bpoly anchors are missing or beyond snap tolerance;
- the deformed seaward-chain frame cannot create a seed-containing water component;
- the offshore arc crosses GSHHS land away from its endpoints;
- the open arc is not present on the accepted wet-domain exterior;
- the seed is not inside the accepted wet domain.

## Topology Method

The default workflow is GSHHS-first:

1. Load `region_bpoly.json` and `offshore_boundary_artifacts.json`.
2. Fetch or load GSHHS `land_polygons` and `coastline_lines`.
3. Project bpoly, coastline, arcs, and land polygons to a local UTM CRS.
4. Find coastline/bpoly intersection anchors on the two bpoly sides adjacent to the selected offshore side.
5. Split the adjacent bpoly sides at those anchors and build the seaward control chain through the two offshore-side corners.
6. Generate smooth fixed-endpoint arc candidates by bowing/deforming that full seaward chain along the offshore azimuth.
7. Build a closed deformed bpoly frame from the selected open arc plus the non-seaward bpoly path between anchors.
8. Compute `deformed_frame - GSHHS land_union`, then choose the seed-containing water component.
9. Classify `open_boundary_arc`, GSHHS-overlapping `land_boundary_arcs`, remaining `frame_clip_boundary_arcs`, and `island_holes`.
10. Write classified boundary layers and visual-review artifacts.
11. Automatically build the continuous model exterior loop, classify exterior segments as open/land/frame/unclassified boundary, and convert wet-domain interior rings into island polygons.

Use `--topology-mode iterative-raster --coastline-source cusp-legacy` only for legacy CUSP/debug cases. Do not make CUSP the normal topology source until a later refinement workflow explicitly controls it as a local-detail overlay.

Read `references/oceanmesh2d_rpw2019_notes.md` before changing shoreline classification, island filtering, or seeded wet-domain extraction. Treat the local OceanMesh2D snapshot as a method reference only; do not copy GPL MATLAB code into this skill implementation without explicit licensing review.

## Validation

From the skill folder:

```powershell
python scripts/selftest_bdry_arc.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
