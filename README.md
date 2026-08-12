# Agent Skill Development Catalog

This folder is the planning and staging area for the FVCOM agent skillset. The
active high-level structure is `skill_catalog/`, which organizes future skills
by capability family rather than by the older broad project-stage folders.

## Catalog Layout

- `common-core/`: shared FVCOM support utilities, currently `fvcom-common`.
- `external-data-connectors/`: source-specific data acquisition and conversion
  capabilities, such as HYCOM, NOAA CO-OPS, USGS, CBOFS, DBOFS, SSCOFS, NYOFS,
  SJROFS, CFSv2, GloFAS, GSHHS, CUDEM, CUSP, NHD/NHM river products, and
  usSEABED.
- `forcing-builders/`: tools that assemble FVCOM-ready forcing products from
  source data or local inputs.
- `grid-generation/`: regional-domain, boundary-arc, coastline-topology, and
  future mesh/refinement skills for FVCOM preprocessing.
- `memory-control/`: project-memory workflow skills, currently
  `brain-dumping` and `brain-refreshing`.
- `model-execution-hpc/`: execution and HPC bridge skills for Kestrel,
  Constance, and the PNNL cloud VM, including Codex and Copilot-facing variants
  where staged.
- `model-analysis/`: active structured-grid POM, staggered-grid ROMS, and
  sparse curvilinear EFDC map post-processing, plus staged future
  scientific-analysis work.

## Installing Skills

This catalog is organized for Codex-style skill systems where each installable
skill is a folder with `SKILL.md` at its root. A compatible harness should copy
individual skill folders, not the whole capability-family folder, into its local
skill directory.

For Codex, the usual install target is:

- Windows: `%USERPROFILE%\.codex\skills\<skill-name>\`
- macOS/Linux: `${CODEX_HOME:-$HOME/.codex}/skills/<skill-name>/`

When installing a skill, preserve the entire folder structure, including
`SKILL.md`, `agents/`, `scripts/`, `references/`, and any bundled helper files.
Some skills require Python packages, remote credentials, or local data sources at
runtime; installing the skill only makes the workflow instructions available.

Install all cataloged skills from the repository root on Windows PowerShell:

```powershell
$dest = Join-Path $env:USERPROFILE ".codex\skills"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Get-ChildItem .\skill_catalog -Directory | ForEach-Object {
  Get-ChildItem $_.FullName -Directory | Where-Object {
    Test-Path (Join-Path $_.FullName "SKILL.md")
  } | ForEach-Object {
    $target = Join-Path $dest $_.Name
    if (Test-Path $target) {
      Remove-Item -LiteralPath $target -Recurse -Force
    }
    Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse
  }
}
```

Install all cataloged skills from macOS/Linux shell:

```bash
dest="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$dest"
for skill in skill_catalog/*/*; do
  if [ -f "$skill/SKILL.md" ]; then
    rm -rf "$dest/$(basename "$skill")"
    cp -R "$skill" "$dest/$(basename "$skill")"
  fi
done
```

Install one individual skill on Windows PowerShell:

```powershell
$dest = Join-Path $env:USERPROFILE ".codex\skills"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$target = Join-Path $dest "brain-refreshing"
if (Test-Path $target) {
  Remove-Item -LiteralPath $target -Recurse -Force
}
Copy-Item -LiteralPath .\skill_catalog\memory-control\brain-refreshing -Destination $target -Recurse
```

Install one individual skill on macOS/Linux shell:

```bash
dest="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$dest"
rm -rf "$dest/brain-refreshing"
cp -R skill_catalog/memory-control/brain-refreshing "$dest/brain-refreshing"
```

If the Codex system skill validator is available, validate an installed or
catalog skill with:

```powershell
python "$env:USERPROFILE\.codex\skills\.system\skill-creator\scripts\quick_validate.py" .\skill_catalog\memory-control\brain-refreshing
```

After copying skills, restart or refresh the agent harness so it reloads the
available skill list.

### Agent Install Prompts

Use this prompt when asking a Codex-like agent to install the whole skillset:

```text
Install all Codex-compatible skills from this repository. Treat every folder
matching skill_catalog/<family>/<skill-name>/SKILL.md as one skill package. Copy
each package into the local Codex skill directory as <skill-name>, preserving
all subfolders and files. Validate that each copied package has SKILL.md at the
package root, then list the installed skill names.
```

Use this prompt when asking an agent to install one skill:

```text
Install only skill_catalog/<family>/<skill-name> as a Codex-compatible skill.
Copy that folder into the local Codex skill directory as <skill-name>,
preserving SKILL.md and any agents, scripts, references, or bundled assets.
Validate the copied skill if a local validator is available, then report the
installed path.
```

Use this prompt when adapting a skill to a non-Codex but similar skill system:

```text
Adapt this skill package for a Codex-like skill system. Preserve the SKILL.md
instructions as the primary activation and workflow document. Keep bundled
scripts, references, and agent metadata attached to the same skill package.
Only change packaging metadata required by the target harness; do not rewrite
scientific workflow rules or remove validation guidance.
```

## Current Development State

The catalog is no longer just a planning skeleton; it now contains several
usable skill families at different maturity levels:

- `external-data-connectors/` entries are maintained as installable skills with
  `SKILL.md` metadata, agent UI metadata, estimate-first routing hooks where
  appropriate, and downloaded-data health checks. The connector set now includes
  HYCOM, NOAA CO-OPS, CFSv2, CUDEM, CUSP, GSHHS, NHD/NHM river tools, USGS
  rivers, usSEABED, `glofas-data-fetcher`, and the AWS-primary
  `cbofs-fetcher`, `dbofs-fetcher`, `sscofs-fetcher`, `nyofs-fetcher`, and
  `sjrofs-fetcher` connectors. The five OFS connectors use reviewed v2 plans, anonymous NOAA
  access, and model-safe NCEI long-term fallback for supported historical
  records when operational AWS coverage is incomplete.
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
  continuous model-boundary-loop package builder. `topobathy-flownet` remains a
  standalone reusable drainage/thalweg analysis, while
  `fvcom-grid-generation` now derives its hydraulic skeleton directly from the
  wet polygon, solid boundary geometry, and bathymetry during mesh-size
  construction and can lock that intent for controlled clean-room/Gmsh
  generator portfolios without changing the production default.
- `memory-control/` now contains the two active HTML project-memory workflow
  skills: `brain-dumping` for durable session memos and `brain-refreshing` for
  workspace reorientation before continuing work.
- `model-analysis/` now includes the active `pom-map-postprocessing`,
  `pom-movie-postprocessing`, `roms-map-postprocessing`, and
  `roms-movie-postprocessing` skills, together with
  `efdc-map-postprocessing` for sparse curvilinear EFDC grids. Script-only
  FVCOM folders without `SKILL.md` remain reference material rather than
  installable skills.
- `common-core/` and `forcing-builders/` remain less mature catalog families and
  should be expanded only through explicit skill development work.

See `../Memory/memo_v003.html` for the planning rationale and the script mapping
from the original staging folders into the catalog.
