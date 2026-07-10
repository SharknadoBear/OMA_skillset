---
name: fvcom-grid-generation
description: Generate FVCOM-ready SMS 2DM meshes from legacy or adaptive fvcom-bdry-arc packages and CUDEM/NBS/CRM/ETOPO bathymetry using clean-room constrained Delaunay refinement. Use when Codex needs explicit boundary-chain ingestion, adaptive nearshore-to-offshore size fields, variable-density seeding, ordered OBC nodestrings, FVCOM mesh QA, or Delaware-style grid smoke tests; standalone OceanMesh-style cleanup remains available for research.
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
- End normal generation at the generation-time smoothed mesh. Do not run post-generation cleanup implicitly.
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
- `--postprocess-profile`, compatibility default `none`; non-`none` integrated requests are rejected with standalone-tool guidance.
- `--land-spacing-m`, `--open-spacing-m`, `--gradation`, `--max-interior-points`, and `--size-field-max-cells` retain their legacy meanings.

Adaptive mode propagates per-boundary-node target sizes with the gradation lower envelope, removes the blanket shallow-water 2 km cap, creates deterministic quadtree interior seeds, and avoids low-angle insertion when local edges are already undersized. Sample triangle targets in vector batches and repeat constraint recovery after the final smoothing/retriangulation; acceptance is based on the delivered mesh, not the pre-refinement recovery state.

## Standalone Tools

- `scripts/analyze_mesh_quality.py`: analyze an existing `.2dm` without altering it.
- `scripts/postprocess_fvcom_mesh.py`: run `rpw2019` or `projection-medium` cleanup explicitly for research.
- `scripts/compare_mesh_quality.py`: compare any two compatible quality JSON documents.

## Normal Outputs

- `fvcom_grid.2dm`
- `fvcom_grid_manifest.json`
- `mesh_quality.json`
- `boundary_nodes.geojson`
- `size_field.nc` and `size_field.png`
- `mesh_nodes_elements.gpkg`
- `mesh_quality_elements.gpkg`
- `mesh_review_map.png`
- progress JSON/JSONL artifacts

The v3 manifest records `postprocess.enabled: false`. Cleanup-specific preclean, history, postclean-boundary, and comparison artifacts are written only by standalone tools.

## Acceptance

Require a successful 2DM roundtrip, finite positive depths, positive-area elements, ordered OBC pairs, complete constraints, one manifold component, and valid exterior/island loops. For adaptive OBCs require 95th-percentile `L/h <= 1.55` and maximum `L/h <= 2`. Retain artifacts with `needs_review` when geometric or FVCOM physics gates fail.

## Validation

```powershell
python scripts/selftest_fvcom_grid.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
