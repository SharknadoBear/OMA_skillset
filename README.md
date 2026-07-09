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
- `memory-control/`: project-memory workflow skills, currently
  `brain-dumping` and `brain-refreshing`.
- `model-execution-hpc/`: execution and HPC bridge skills for Kestrel,
  Constance, and the PNNL cloud VM, including Codex and Copilot-facing variants
  where staged.
- `model-analysis/`: reserved for future post-processing and scientific
  analysis skills.

The previous reserved `quality-control/` and `memory-governance/` catalog
families have been removed. Quality checks should live inside each reviewed
skill's own scripts and acceptance rules. Project/lab protocol also does not
allow agents to proactively modify deployed skills during screening or
deployment; skill edits must be explicit development work, validated, and
committed through the catalog workflow.

## Current Development State

The catalog is no longer just a planning skeleton; it now contains several
usable skill families at different maturity levels:

- `external-data-connectors/` entries are maintained as installable skills with
  `SKILL.md` metadata, agent UI metadata, estimate-first routing hooks where
  appropriate, and downloaded-data health checks. The connector set now includes
  HYCOM, NOAA CO-OPS, CBOFS/DBOFS, CFSv2, CUDEM, CUSP, GSHHS, NHD/NHM river
  tools, USGS rivers, usSEABED, and the newly cataloged
  `glofas-data-fetcher`.
- `model-execution-hpc/kestrel-hpc` is a robust operational bridge skill, not
  merely a copied placeholder. It records Bear's Kestrel account context,
  required SSH MAC option, Password+OTP handling rules, Slurm job inspection and
  monitoring patterns, upload/download guidance, and a reusable local Paramiko
  bridge workflow for multi-command sessions. It should be treated as the
  primary Kestrel access skill for controlled compile, transfer, job-monitoring,
  and compact-output retrieval tasks.
- `model-execution-hpc/constance-hpc` and `model-execution-hpc/cloudvm-bridge`
  are also staged as execution/connectivity skills, with Copilot sibling folders
  retained where migration work has been performed.
- `grid-generation/` now contains active FVCOM preprocessing skills rather than
  only future placeholders. `fvcom-region-bpoly` is the first-stage regional
  domain selector, and `fvcom-bdry-arc` is the second-stage boundary-arc and
  continuous model-boundary-loop package builder. `fvcom-grid-generation`
  remains the downstream mesh-generation skill area.
- `memory-control/` now contains the two active HTML project-memory workflow
  skills: `brain-dumping` for durable session memos and `brain-refreshing` for
  workspace reorientation before continuing work.
- `common-core/`, `forcing-builders/`, and `model-analysis/` remain less mature
  catalog families and should be expanded only through explicit skill
  development work.

See `../Memory/memo_v003.html` for the planning rationale and the script mapping
from the original staging folders into the catalog.
