---
name: fvcom-grid-generation
description: Generate FVCOM-ready SMS 2DM meshes from legacy or adaptive fvcom-bdry-arc packages and CUDEM/NBS/CRM/ETOPO bathymetry using clean-room constrained refinement, offshore-efficient sizing, regional spring relaxation, aggressive local topology conditioning, target-aware area-transition conditioning, and boundary-aware thin-triangle repair. Use when Codex needs explicit boundary-chain ingestion, adaptive nearshore-to-offshore size fields, variable-density seeding, regional mesh conditioning, hard FVCOM valence repair, ordered OBC nodestrings, FVCOM mesh QA, or small synthetic grid smoke tests; standalone broad OceanMesh-style cleanup remains available for research.
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
- End normal adaptive generation after guarded shape relaxation, first-pass thin repair, aggressive local topology conditioning, target-aware area-transition relaxation, and a terminal constraint audit. Do not run the broad legacy postprocessor implicitly.
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
- `--conditioning-profile auto|guarded-v1|aggressive-local-v2|none`; `auto` promotes adaptive packages to `aggressive-local-v2` and retains guarded legacy behavior.
- `--aggressive-conditioning-rounds`, `--aggressive-max-prunes-per-round`, and `--aggressive-max-valence-repairs-per-round`; bound repeated local edit/relax transactions.
- Standalone `repair_high_valence.py` additionally accepts `--max-valence-flip-batch` (default 64), `--max-valence-cluster-merges-per-round` (default 25), `--max-valence-l-over-h-count-increase` (default 0), and repeatable `--only-node-id-1based` targeting. Independent legal flips share one audit; adjacent zipper violations are attempted as one simple interior cavity before sequential single-node work. Keep the (L/h) trade budget at zero unless a documented hard-gate closure justifies it.
- `--aggressive-boundary-edit-policy kind-aware-envelope|split-only|none`; controls whether a non-hard boundary vertex may be removed inside a strict kind-specific shape/area envelope.
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
4. In `aggressive-local-v2`, repeat local transactions in this order: remove target-redundant degree-3/4 interior vertices; collapse a short interior edge shared by a superthin pair to its midpoint; for a boundary superthin element, split its longest side when that side is a protected boundary arc, otherwise consider removal of a non-hard boundary vertex inside the boundary envelope; repair every valence above eight with batched legal flips, connected zipper-cavity merges, or sequential cavity removal with distributed Steiner nodes. If an admissible boundary removal transfers overload to its neighbor, stabilize the affected boundary strip inside the same transaction. Follow every accepted topology edit with tightly restricted two-ring spring relaxation.
5. Apply `area-transition-relax-v1` to the worst excessive adjacent-area pairs one patch at a time. Always consider raw area change above 0.50; preempt inside steep target-size bands only when raw and target-normalized area mismatch are both excessive. Re-sample the Eulerian target field before every outer patch.
6. Audit protected chains, ordered OBC pairs, positive areas, manifold components, surviving original boundary coordinates, hard anchors, area-transition tails, and `L/h`. Roll back any patch or whole stage that regresses its stage baseline. Treat any remaining valence above eight as a hard FVCOM gate failure.
7. Sample bathymetry at the delivered nodes, run FVCOM QA, and write the 2DM plus an edit ledger and terminal boundary lineage.

Hard anchors and all boundary nodes not explicitly handled by the kind-aware edit protocol remain fixed. Boundary splitting follows the existing arc exactly; boundary-vertex removal is forbidden at hard anchors and is accepted only inside the configured chord-deviation and cumulative domain-area envelopes. Treat unresolved boundary- or topology-imposed thin/transition defects as evidence for `needs_review`; never flatten the scientific target-size field or invoke a global Delaunay rebuild to force a pass. See `references/local_topology_conditioning.md` for the mathematics and transaction gates.

## Standalone Tools

- `scripts/analyze_mesh_quality.py`: analyze an existing `.2dm` without altering it.
- `scripts/relax_mesh_region.py`: apply the boundary-fixed spring solver to automatic defect patches or a requested lon-lat bounding box.
- `scripts/repair_thin_triangles.py`: apply transactional local flips/splits and patch relaxation while preserving the model boundary and OBC order.
- `scripts/condition_mesh_local.py`: apply the complete `aggressive-local-v2` protocol to an existing 2DM, with optional adaptive boundary metadata and size field.
- `scripts/repair_high_valence.py`: run only the hard valence-repair branch; returns a nonzero status while any node valence exceeds eight.
- `scripts/diagnose_high_valence.py`: inventory every valence violation in one topology scan, classify its likely repair route, write CSV/GeoJSON/JSON, and plot domain-wide plus graph-local diagnostic maps; supply a conditioning report to plot rejected cases first.
- `scripts/prune_redundant_vertices.py`: run only target-aware degree-3/4 interior-vertex pruning.
- `scripts/postprocess_fvcom_mesh.py`: run `rpw2019` or `projection-medium` cleanup explicitly for research.
- `scripts/compare_mesh_quality.py`: compare any two compatible quality JSON documents.

## Normal Outputs

- `fvcom_grid.2dm`
- `fvcom_grid_manifest.json`
- `mesh_quality.json`
- `mesh_conditioning.json`
- `mesh_edit_ledger.json` (operation, source-node lineage, and local edit evidence)
- `boundary_nodes.geojson` (input boundary-node package)
- `delivered_boundary_nodes.geojson` (terminal constraint chains, including any recovery nodes)
- `size_field.nc` and `size_field.png`
- `mesh_nodes_elements.gpkg`
- `mesh_quality_elements.gpkg`
- `mesh_review_map.png`
- progress JSON/JSONL artifacts

The v6 manifest records shape, guarded thin repair, aggressive local topology conditioning, area-transition conditioning, hard valence status, and terminal lineage separately from `postprocess.enabled: false`. Broad-cleanup preclean, history, postclean-boundary, and comparison artifacts remain standalone-only.

## Acceptance

Require a successful 2DM roundtrip with matching connectivity and OBC order, finite positive depths, positive-area elements, complete constraints, one manifold component, exact coordinates for surviving original boundary nodes, exact hard-anchor survival, sub-centimeter text-serialization shifts, and valid exterior/island loops. Conditioning must not materially regress controlled lower-tail metrics, stage-baseline `L/h` (explicit 0.1% numerical tolerance), or area-transition defect counts. Require true vertex-neighbor valence `<=8` for FVCOM readiness. For adaptive OBCs require 95th-percentile `L/h <= 1.55` and maximum `L/h <= 2`. Retain artifacts with `needs_review` when a legal guarded edit cannot close a gate.

## Validation

```powershell
python scripts/selftest_fvcom_grid.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
