# Agent Skill Development Catalog

This folder is the planning and staging area for the FVCOM agent skillset. The
active high-level structure is `skill_catalog/`, which organizes future skills
by capability family rather than by the older broad project-stage folders.

## Catalog Layout

- `common-core/`: shared FVCOM support utilities, currently `fvcom-common`.
- `external-data-connectors/`: source-specific data acquisition and conversion
  capabilities, such as HYCOM, NOAA CO-OPS, USGS, CBOFS, DBOFS, CFSv2, and
  usSEABED, including the NHM river fetcher.
- `forcing-builders/`: tools that assemble FVCOM-ready forcing products from
  source data or local inputs.
- `grid-generation/`: reserved for future mesh and grid refinement skills.
- `model-execution-hpc/`: execution and HPC bridge skills, including the copied
  `kestrel-hpc` skill.
- `model-analysis/`: reserved for future post-processing and scientific
  analysis skills.
- `quality-control/`: reserved for validation, smoke-test, and artifact-checking
  capabilities.
- `memory-governance/`: reserved for project memory, provenance, and governance
  support capabilities.

## Current Development State

Most non-connector catalog folders are not installable skills yet. The
`external-data-connectors/` entries are maintained as installable skills with
`SKILL.md` metadata, agent UI metadata, estimate-first routing hooks, and
downloaded-data health checks. The `model-execution-hpc/kestrel-hpc` folder is
also copied from the existing local Kestrel HPC skill and includes its skill
metadata.

See `../Memory/memo_v003.tex` for the planning rationale and the script mapping
from the original staging folders into the catalog.
