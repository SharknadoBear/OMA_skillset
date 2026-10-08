# Agent Skill Development Catalog

This folder is the planning and staging area for the OMA coastal modeling skillset, including FVCOM and REMORA. The
active high-level structure is `skill_catalog/`, which organizes future skills
by capability family rather than by the older broad project-stage folders.

## Catalog Layout

- `simulation-meta/`: complete FVCOM workflow orchestration, build,
  namelist, stability/run-control, and benchmark skills. `fvcom-simulation` is
  the scriptless scenario router; specialist packages own all mechanics.
- `external-data-connectors/`: source-specific data acquisition and conversion
  capabilities, such as the model-neutral `hycom-fetcher`, `argo-fetcher`, and the
  NCEI-first, era-routing `cfsv2-fetcher`/`cfsr-fetcher` pair and resilient
  multi-mirror `hrrr-fetcher`,
  ERA5 atmosphere, GLORYS12 ocean reanalysis, NOAA WAVEWATCH III waves,
  NOAA CO-OPS, USGS, CBOFS, DBOFS, SSCOFS, NYOFS, SJROFS, GloFAS, GSHHS,
  CUDEM, CUSP, NHD/NHM river products, and
  usSEABED, including model-neutral TPXO9v5 harmonic extraction.
- `external-tool-connectors/`: instructions for third-party scientific tools
  that are installed separately, currently UTide.
- `forcing-builders/`: tools that assemble FVCOM-ready forcing products from
  source data or local inputs, including validated boundary water-level,
  temperature/salinity, and modular surface-flux forcing.
- `grid-generation/`: regional-domain, boundary-arc, coastline-topology, and
  mesh/refinement skills for FVCOM, plus independent scientific region planning
  and structured single-level grid generation for REMORA.
- `memory-control/`: project-memory workflow skills, currently
  `brain-dumping` and `brain-refreshing`.
- `workspace-bridging/`: skills that bridge local workspaces to Kestrel,
  Expanse, Constance, and configured cloud VM execution environments, including
  Codex and Copilot-facing variants where staged.
- `visual-analysis/`: active structured-grid POM, staggered-grid ROMS, and
  sparse curvilinear EFDC map and movie post-processing, FVCOM tidal
  validation, and `fvcom-velocity-tracer-movie` offline HTML current shading
  and finite-age particle trails with independent visual-speed controls and offline snapshot GIF export, plus
  staged scientific-analysis work.

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
  Get-ChildItem $_.FullName -Filter "SKILL.md" -File -Recurse | ForEach-Object {
    $skill = $_.Directory
    $target = Join-Path $dest $skill.Name
    if (Test-Path $target) {
      Remove-Item -LiteralPath $target -Recurse -Force
    }
    Copy-Item -LiteralPath $skill.FullName -Destination $target -Recurse
  }
}
```

Install all cataloged skills from macOS/Linux shell:

```bash
dest="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$dest"
find skill_catalog -type f -name SKILL.md -print0 |
while IFS= read -r -d '' skill_file; do
  skill="$(dirname "$skill_file")"
  rm -rf "$dest/$(basename "$skill")"
  cp -R "$skill" "$dest/$(basename "$skill")"
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
containing a SKILL.md anywhere beneath skill_catalog/ as one skill package. Copy
each package into the local Codex skill directory using its folder name,
preserving all subfolders and files. Validate that each copied package has
SKILL.md at the package root, then list the installed skill names.
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

The catalog contains usable skill families at different maturity levels.

The REMORA gridding workflow has two independently invocable skills under
`skill_catalog/grid-generation/`; install both as sibling skill folders:

| Skill | Responsibility | Validation |
| --- | --- | --- |
| `remora-region-bpoly` | Four-sided scientific regions, required features, optional parent/child interests, maps and recorded visual review | Polygon/hierarchy and artifact-integrity tests in `scripts/selftest.py` |
| `remora-grid-generation` | Rotated orthogonal footprint, staggered coordinates/metrics, source bathymetry, mask correction, log-depth smoothing, vertical diagnostics, CDF5 NetCDF and standardized case delivery | Numerical/error-path tests, combined two-skill CLI test and a separate executable reader checker |

Grid generation uses Python with prebuilt dependencies and requires no local
C/C++ compiler. A complete request invokes the polygon stage and returns for
visual review before fitting and building the grid. Existing reviewed region
packages are also accepted. Region hierarchy describes future refinement
interests; v1 generates one root grid and does not implement numerical nesting.
The Delaware Bay release was exercised at 500 m with 40 layers and checked
against a pinned REMORA executable on Kestrel using zero-step initialization.
That reader check is distinct from scientific time-integration validation.
Keep regional data, private execution receipts, and memory in the project
workspace, outside these reusable packages.

The October 2026 additions are standalone, model-neutral acquisition packages
under `skill_catalog/external-data-connectors/`:

| Skill | Source and supported products | Runtime and validation |
| --- | --- | --- |
| `era5-fetcher` | CDS ERA5 hourly 10 m wind and mean sea-level pressure, bounded by a geographic box or ADCIRC mesh | Python 3.11.9; frozen requirements; offline `scripts/selftest_era5.py --output REPORT.json` |
| `glorys-fetcher` | Copernicus GLORYS12 daily/monthly ocean fields, including water level, potential temperature, salinity and horizontal currents | Python 3.11; Copernicus Marine Toolbox 2.5.0 and frozen requirements; offline `scripts/selftest_glorys.py` |
| `noaa-ww3-fetcher` | NOAA historical `multi_1` WAVEWATCH III monthly gridded fields and selected-station directional spectra | Python 3.10+ with requests, NumPy, rasterio/GDAL and netCDF4; local-fixture `scripts/test_fetcher.py` |

Each package preserves bounded requests, source provenance, verified resume/cache
behavior, native time/coordinate/mask conventions, and independent health checks.
ERA5 requires local CDS credentials and accepted dataset licences; GLORYS uses
local Copernicus Marine credentials. NOAA WW3 uses the public historical archive.
Keep credentials, downloads and run evidence in the project outside the skill.
Offline tests validate package behavior; authenticated live acquisition and source
availability must be checked for each requested run. See each package's `SKILL.md`
and request reference for its exact scope and invocation.

- `external-data-connectors/` entries are maintained as installable skills with
  `SKILL.md` metadata, agent UI metadata, estimate-first routing hooks where
  appropriate, and downloaded-data health checks. The connector set now includes
  `argo-fetcher` for native core/B/S GDAC profiles, `hycom-fetcher` with thin
  Codex/Hermes variants, NOAA CO-OPS,
  NCEI-first `cfsv2-fetcher`/`cfsr-fetcher` with automatic era routing,
  `hrrr-fetcher` for AWS-first, message-ranged CONUS and Alaska analysis/forecast fields,
  CUDEM, CUSP, GSHHS, NHD/NHM river tools, USGS
  rivers, usSEABED, `glofas-data-fetcher`, and the AWS-primary
  `cbofs-fetcher`, `dbofs-fetcher`, `sscofs-fetcher`, `nyofs-fetcher`, and
  `sjrofs-fetcher` connectors, plus `tpxo9v5-data-fetcher` for registered
  model-neutral harmonic subsets and interpolation. The five OFS connectors use
  reviewed v2 plans, anonymous NOAA access, and model-safe NCEI long-term fallback
  for supported historical records when operational AWS coverage is incomplete.
  HYCOM, CFSv2, and CFSR requests use bounded transfer probes and persistent JSON
  progress; conservative estimates of ten minutes or longer open a localhost
  HTML waitbar automatically.
- `workspace-bridging/kestrel-hpc` is a robust operational bridge skill, not
  merely a copied placeholder. It uses runtime-supplied account and host context,
  preserves the required SSH MAC option, protects interactive credentials, and supports Slurm inspection and
  monitoring patterns, upload/download guidance, and a reusable local Paramiko
  bridge workflow for multi-command sessions. It should be treated as the
  primary Kestrel access skill for controlled compile, transfer, job-monitoring,
  and compact-output retrieval tasks.
- `simulation-meta/fvcom-simulation` is the scriptless parent workflow for
  accepted-then-fresh tide-only and benchmark studies. Its sibling skills own
  Kestrel build lineage, FVCOM 4.3.1 namelists, immutable stability attempts,
  and rank/node Pareto analysis. `fvcom-common` was removed because it had no
  skill entrypoint and duplicated the TPXO builder utilities byte-for-byte.
- `workspace-bridging/expanse-hpc` mirrors the named-session JSON bridge
  architecture for SDSC Expanse, supports password-or-agent authentication
  followed by TOTP, and documents Expanse-specific Slurm, Lmod, project,
  Lustre, node-local storage, and resource-selection rules. Its Copilot sibling
  reuses the same helper package.
- `workspace-bridging/constance-hpc` and `workspace-bridging/cloudvm-bridge`
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
  `fvcom-regional-grid-refinement` adds deterministic element-aligned local
  refinement for existing FVCOM/SMS meshes while preserving protected node
  identities, boundary chords, bathymetry lineage, and preconfiguration
  contracts.
- `memory-control/` now contains the two active HTML project-memory workflow
  skills: `brain-dumping` for durable session memos and `brain-refreshing` for
  workspace reorientation before continuing work.
- `visual-analysis/` now includes the active `pom-map-postprocessing`,
  `pom-movie-postprocessing`, `roms-map-postprocessing`, and
  `roms-movie-postprocessing` skills, together with
  `efdc-map-postprocessing` and `efdc-movie-postprocessing` for sparse
  curvilinear EFDC grids. Script-only FVCOM folders without `SKILL.md` remain
  reference material rather than installable skills.
- `external-tool-connectors/u-tide-tool-instruction` documents
  public UTide harmonic-analysis and reconstruction workflows without vendoring
  the external package.
- `forcing-builders/fvcom-boundary-waterlevel-forcing` maps an existing combined
  NetCDF or CSV water-level record onto FVCOM OBC nodes, writes exact redundant
  FVCOM time representations, and produces mandatory spectral, sample-series,
  and complete-boundary Hovmöller validation.
- `forcing-builders/fvcom-boundary-ts-forcing` packages an existing sigma-ready,
  per-node temperature/salinity NetCDF as an FVCOM T/S OBC file, performs
  auditable within-axis missing-value repair, enforces a zero-NaN gate, and
  produces time-series, time-depth, Hovmöller, and vertical-transect QA.
- `forcing-builders/fvcom-surface-fluxes-forcing` packages any selected subset
  of prepared wind, direct or bulk heat, freshwater, and atmospheric-pressure
  matrices on structured or FVCOM-native grids, applies source-aware sign and
  unit gates, writes safe combined or split files, and produces mandatory
  scientific QA plus a namelist fragment.
- Remaining preliminary `forcing-builders/` entries should be expanded only
  through explicit skill development work.

See `../Memory/memo_v003.html` for the planning rationale and the script mapping
from the original staging folders into the catalog.

The gridding skill now applies area-weighted pair corrections to natural-log depth and reports volume changes without global rescaling. Its `finalize` and `validate-case` entry points provide a fixed final NetCDF, six ordered review panel groups, source/attempt provenance, and independent agent, human and executable-reader statuses. The numerical update passed 23 unit/error tests, combined CLI checks, and a separately regenerated Delaware visual regression; the previous executable reader receipt does not apply to this changed grid.
