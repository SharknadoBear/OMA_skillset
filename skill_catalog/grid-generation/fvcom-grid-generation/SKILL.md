---
name: fvcom-grid-generation
description: Generate FVCOM-ready SMS 2DM meshes from legacy or adaptive fvcom-bdry-arc packages and CUDEM/NBS/CRM/ETOPO bathymetry using clean-room constrained refinement, offshore-efficient sizing, regional spring relaxation, target-aware area-transition conditioning, and boundary-preserving thin-triangle repair. Use when Codex needs explicit boundary-chain ingestion, adaptive nearshore-to-offshore size fields, variable-density seeding, regional mesh conditioning, ordered OBC nodestrings, FVCOM mesh QA, or small synthetic grid smoke tests; standalone broad OceanMesh-style cleanup remains available for research.
---

# fvcom-grid-generation

Use this skill as the third OMA gridding step:

```text
fvcom-region-bpoly -> fvcom-bdry-arc -> cudem-bathy -> fvcom-grid-generation
```

## Core Rules

- Reuse upstream domain and boundary artifacts; do not redesign the region here.
- Prefer an adaptive boundary-resolution manifest when supplied. Otherwise preserve the legacy loop workflow.
- Keep full bathymetry for final node sampling and bound only the in-memory size-field grid.
- In adaptive mode, let explicit OBC/boundary targets control offshore resolution; apply bathymetric-gradient refinement only in the configured coastal/estuarine influence zone unless the user explicitly selects global behavior.
- End normal generation after guarded shape relaxation, thin-triangle repair, target-aware area-transition relaxation, and a terminal constraint audit. Do not run the broad legacy postprocessor implicitly.
- Keep depths finite and positive down and treat OceanMesh2D GPL material as a method reference only.

## Primary Workflow

Adaptive package:

```powershell
python scripts/run_fvcom_grid.py --bdry-arc-manifest bdry_arc_manifest.json --boundary-loops-gpkg model_boundary_loops.gpkg --boundary-resolution-manifest boundary_resolution_manifest.json --boundary-resolution-profile adaptive-coastal-v1 --bathy-nc bathy.nc --run-dir runs/case --name case --mode test --postprocess-profile none
```

Legacy packages continue to use `--boundary-loops-gpkg` without a resolution manifest.

Important controls:

- `--boundary-resolution-profile legacy|adaptive-coastal-v1`, default `legacy`.
- `--boundary-resolution-manifest`, optional; explicit nodes and chains take precedence over legacy densification.
- `--bathy-gradient-policy auto|global|coastal|off`; `auto` means `coastal` for adaptive packages and preserves `global` legacy behavior.
- `--coastal-gradient-distance-m`, default 25 km; controls where bathymetric slope may refine an adaptive mesh.
- `--regional-spring-relaxation|--no-regional-spring-relaxation`; normal generation applies one guarded, defect-selected spring-equilibrium stage by default.
- `--thin-triangle-repair|--no-thin-triangle-repair`; normal generation applies local protected-edge-safe flips/splits and patch relaxation by default.
- `--area-transition-relaxation|--no-area-transition-relaxation`; normal generation applies sequential target-aware spring patches after thin repair.
- `--area-transition-max-patches`, default 12; bounds accepted local transition patches. The raw adjacent-area trigger defaults to `0.50`, equivalent to an area ratio of two.
- `--postprocess-profile`, compatibility default `none`; non-`none` integrated requests are rejected with standalone-tool guidance.
- `--land-spacing-m`, `--open-spacing-m`, `--gradation`, `--max-interior-points`, and `--size-field-max-cells` retain their legacy meanings.

Adaptive mode propagates per-boundary-node target sizes with the gradation lower envelope, removes the blanket shallow-water 2 km cap, suppresses offshore bathymetric-gradient over-refinement, creates deterministic quadtree interior seeds, and avoids low-angle insertion when local edges are already undersized. Persist boundary, slope, coastal-mask, and coastal-distance attribution in the size-field NetCDF and report.

## Generation-Time Conditioning

Use this order after constrained seeding/refinement:

1. Recover all protected land, island, frame, and open-boundary edges.
2. Apply `spring-relax-v1` once to automatically selected poor-element patches. Keep physical boundary nodes fixed, keep connectivity fixed, and accept only backtracked force steps that preserve positive areas and do not regress controlled quality tails.
3. Apply `thin-repair-v1` to residual severe elements. Try legal nonprotected edge flips, then budgeted long interior-edge splits; relax only the edited patch and its graph halo.
4. Apply `area-transition-relax-v1` to the worst excessive adjacent-area pairs one patch at a time. Always consider raw area change above 0.50; preempt inside steep target-size bands only when raw and target-normalized area mismatch are both excessive. Re-sample the Eulerian target field before every outer patch.
5. Audit protected chains, ordered OBC pairs, positive areas, manifold components, exact original boundary coordinates, area-transition tails, and `L/h`. Roll back any patch or whole stage that regresses its stage baseline.
6. Sample bathymetry at the delivered nodes, run FVCOM QA, and write the 2DM.

Treat unresolved boundary- or topology-imposed thin/transition defects as evidence for `needs_review`; never flatten the scientific target-size field or move/delete a protected boundary to force a pass. Do not call a global Delaunay rebuild from any conditioning stage.

## Standalone Tools

- `scripts/analyze_mesh_quality.py`: analyze an existing `.2dm` without altering it.
- `scripts/relax_mesh_region.py`: apply the boundary-fixed spring solver to automatic defect patches or a requested lon-lat bounding box.
- `scripts/repair_thin_triangles.py`: apply transactional local flips/splits and patch relaxation while preserving the model boundary and OBC order.
- `scripts/postprocess_fvcom_mesh.py`: run `rpw2019` or `projection-medium` cleanup explicitly for research.
- `scripts/compare_mesh_quality.py`: compare any two compatible quality JSON documents.

## Normal Outputs

- `fvcom_grid.2dm`
- `fvcom_grid_manifest.json`
- `mesh_quality.json`
- `mesh_conditioning.json`
- `boundary_nodes.geojson` (input boundary-node package)
- `delivered_boundary_nodes.geojson` (terminal constraint chains, including any recovery nodes)
- `size_field.nc` and `size_field.png`
- `mesh_nodes_elements.gpkg`
- `mesh_quality_elements.gpkg`
- `mesh_review_map.png`
- progress JSON/JSONL artifacts

The v5 manifest records shape, thin-repair, and area-transition conditioning separately from `postprocess.enabled: false`. Broad-cleanup preclean, history, postclean-boundary, and comparison artifacts remain standalone-only.

## Acceptance

Require a successful 2DM roundtrip with matching connectivity and OBC order, finite positive depths, positive-area elements, complete constraints, one manifold component, exact original boundary coordinates in projected working space, sub-centimeter text-serialization shifts, and valid exterior/island loops. Conditioning must not materially regress controlled lower-tail metrics, stage-baseline `L/h` (explicit 0.1% numerical tolerance), or area-transition defect counts. For adaptive OBCs require 95th-percentile `L/h <= 1.55` and maximum `L/h <= 2`. Retain artifacts with `needs_review` when protected geometry or fixed topology prevents a legal repair or geometric/FVCOM physics gates remain open.

## Validation

```powershell
python scripts/selftest_fvcom_grid.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
