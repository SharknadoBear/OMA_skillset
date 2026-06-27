---
name: fvcom-grid-generation-copilot
harness: github-copilot
description: GitHub Copilot variant for FVCOM unstructured triangular mesh generation. Use run_in_terminal for scripts, view_image for domain/mesh maps, read_file for quality reports, vscode_askQuestions for review gates. Scripts shared with sibling fvcom-grid-generation/ folder.
---

# FVCOM Grid Generation — GitHub Copilot Harness

## Purpose

Generate FVCOM-ready SMS `.2dm` unstructured triangular meshes from local bathymetry and coastline data with explicit open-boundary nodestrings, quality checks, and visual QA gates.

## Scripts Location

All scripts in sibling folder:

```
Agent_skill_dev/skill_catalog/grid-generation/fvcom-grid-generation/scripts/
```

Key scripts:
- `prepare_coastline_domain.py` — Domain prep with visual review gate
- `record_domain_review.py` — Mark domain review decision
- `generate_coastline_mesh.py` — Gmsh meshing (post-review)
- `generate_from_bathymetry.py` — Quick ellipse workflow (smoke tests)
- `generate_synthetic_mesh.py` — Synthetic test mesh
- `quality_report.py` — Quality metrics for existing 2DM
- `roundtrip_2dm.py` — Parse/rewrite validation

## Copilot Execution Workflow

### Quick Smoke Test

```powershell
# run_in_terminal (mode=sync)
python "Agent_skill_dev\skill_catalog\grid-generation\fvcom-grid-generation\scripts\generate_synthetic_mesh.py" --run-dir Workspace/Grid_preprocessing/runs/copilot_smoke
```

Then:
- `read_file` on `quality_report.json`
- `view_image` on `synthetic_fvcom_grid.png`

### Production Coastline Workflow

#### Step 1: Prepare Domain (Stops for Review)

```powershell
python "Agent_skill_dev\skill_catalog\grid-generation\fvcom-grid-generation\scripts\prepare_coastline_domain.py" bathy.nc cusp_coastline.gpkg --run-dir Workspace/Preprocessing/fvcom-grid-generation/runs/case --name case --bbox W S E N --target-resolution-m 100 --open-boundary-mode auto
```

#### Step 2: Visual Review Gate (Copilot-Specific)

1. `view_image` on `case_domain_review.png` and `case_size_fields.png`
2. `read_file` on `case_domain_visual_review.json` — check review status
3. `vscode_askQuestions` to ask user: "Domain boundary and open boundary look acceptable? [pass/fail/needs_followup]"
4. Record decision:

```powershell
python "...\scripts\record_domain_review.py" --manifest runs/case/case_domain_visual_review.json --decision pass --reviewer copilot-agent --notes "Domain visually verified"
```

#### Step 3: Generate Mesh (Only After Pass)

```powershell
python "...\scripts\generate_coastline_mesh.py" runs/case/case_domain_metadata.json bathy.nc --output-2dm runs/case/case.2dm --quality-json runs/case/case_quality.json
```

#### Step 4: Quality Check

`read_file` on `case_quality.json`. Acceptance criteria:
- min interior angle ≥ 30°
- max interior angle ≤ 130°
- max bathymetric slope ≤ 0.1
- adjacent area-change ≤ 0.5
- max node connectivity ≤ 8
- open-boundary triangles ~normal to boundary

### Ellipse Workflow (Quick/Debug)

```powershell
python "...\scripts\generate_from_bathymetry.py" bathy.nc --output-2dm runs/grid.2dm --quality-json runs/quality.json --offshore-side east
```

### Quality Report on Existing Mesh

```powershell
python "...\scripts\quality_report.py" Resources/Base_C_D_2m_v2_degree_smooth.2dm --output-json runs/quality.json
```

### Roundtrip Validation

```powershell
python "...\scripts\roundtrip_2dm.py" input.2dm --output-2dm output.2dm
```

## Gradation

- Default: `g = 0.15` (conservative)
- Experimental: `g = 0.35` (requires explicit user intent + documented quality review)
- Limiter only reduces coarser cells; never coarsens already-refined cells

## Acceptance Criteria

Accept mesh only when:
- `.2dm` parses successfully after writing
- All wet nodes have finite positive depth
- Triangles are counterclockwise with positive projected area
- Explicit open-boundary `NS` string exists
- Offshore boundary smooth/curved, ordered, visually approved
- No triangle on land or unresolved filtered islands
- Quality failures zero or documented

## Copilot Tool Integration

| Step | Tool |
|------|------|
| Run scripts | `run_in_terminal` (mode=sync) |
| View domain/mesh maps | `view_image` |
| Read quality JSON | `read_file` |
| Ask user for review decision | `vscode_askQuestions` |
| View mesh diagnostic | `view_image` on `*_grid.png` |

## References

Read before changing thresholds:
- `references/fvcom_chapter20_sms_guidance.md` — quality checks, boundary strings
- `references/rpwcw2019_mesh_guidance.md` — size fields, gradation, refinement
