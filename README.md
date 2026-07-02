# Agent Skill Development Catalog

This folder is the planning and staging area for the FVCOM agent skillset. The
active high-level structure is `skill_catalog/`, which organizes future skills
by capability family rather than by the older broad project-stage folders.

## Catalog Layout

- `common-core/`: shared FVCOM support utilities, currently `fvcom-common`.
- `external-data-connectors/`: source-specific data acquisition and conversion
  capabilities, such as HYCOM, NOAA CO-OPS, USGS, CBOFS, DBOFS, CFSv2,
  GloFAS, GSHHS, CUDEM, CUSP, NHD/NHM river products, and usSEABED.
- `forcing-builders/`: tools that assemble FVCOM-ready forcing products from
  source data or local inputs.
- `grid-generation/`: regional-domain, boundary-arc, coastline-topology, and
  future mesh/refinement skills for FVCOM preprocessing.
- `model-execution-hpc/`: execution and HPC bridge skills, including the copied
  `kestrel-hpc` skill.
- `model-analysis/`: reserved for future post-processing and scientific
  analysis skills.

The previous reserved `quality-control/` and `memory-governance/` catalog
families have been removed. Quality checks should live inside each reviewed
skill's own scripts and acceptance rules. Project/lab protocol also does not
allow agents to proactively modify deployed skills during screening or
deployment; skill edits must be explicit development work, validated, and
committed through the catalog workflow.

## Current Development State

Most non-connector catalog folders are not installable skills yet. The
`external-data-connectors/` entries are maintained as installable skills with
`SKILL.md` metadata, agent UI metadata, estimate-first routing hooks where
appropriate, and downloaded-data health checks. The GloFAS connector is now
cataloged as `external-data-connectors/glofas-data-fetcher`. The
`model-execution-hpc/kestrel-hpc` folder is also copied from the existing local
Kestrel HPC skill and includes its skill metadata.

See `../Memory/memo_v003.html` for the planning rationale and the script mapping
from the original staging folders into the catalog.
