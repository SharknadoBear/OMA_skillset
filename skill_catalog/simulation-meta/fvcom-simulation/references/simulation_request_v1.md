# `simulation_request_v1`

```yaml
schema: simulation_request_v1
mode: barotropic_tide | benchmark
profile: paper_barotropic # optional; omit for the ordinary workflow
case_id: string
project_dir: optional absolute or workspace-relative path
grid_policy: accepted_then_fresh | fresh_only
period:
  analysis_start: optional UTC timestamp
  analysis_end: optional UTC timestamp
  spinup_days: optional positive number
physics:
  formulation: three_dimensional_barotropic
  temperature_c: optional number
  salinity_psu: optional number
  sigma_levels: 10 # ten interfaces, nine layers; legacy sigma_layers also counts levels
tpxo:
  constituents: all_available
observations:
  provider: noaa_coops
  station_policy: all_available_inside_wet_domain
  water_level_mapping_policy: strict_inside  # or containing_cell_or_nearest_wet_cell
  current_policy: downward_all_bin_profiles_only
kestrel:
  source_dir: /home/yhuang168/FVCOM_source_agent_test
  account: hindcastra
  partition: auto
resume_from: optional state or attempt token
benchmark: # optional explicit override of the exploratory default
  ranks: [52, 104, 156, 208]
  probe_ranks: [312, 416]
  probe_simulated_hours: 1
  segment_simulated_hours: 24
  repeats: 1
  max_ranks: 416
  extend: false
  production_objective: fastest # or pareto_knee, least_node_hour
```

Dates must be UTC and analysis end is exclusive. Defaults for an unspecified Galveston test are analysis `[2025-04-01T00:00:00Z, 2025-05-01T00:00:00Z)`, seven spin-up days, a forcing start of `2025-03-25T00:00:00Z`, and six-minute forcing/station cadence.

`fresh_only` starts from a new grid and initializes only `fresh_reproduction` scientific branches. It never imports accepted inputs. `accepted_then_fresh` retains the legacy two-case order. The manifest's `grid_order` is authoritative for stage iteration. `forcing/shared_source` means source shared by roles within this project, not between independent projects.

Default new projects to `Workspace/fvcom-simulation/<case_id>`. Explicit and historical roots remain valid. `physics.sigma_layers` is a legacy alias for the number of sigma levels; when both aliases are provided they must agree. Do not change the physical vertical grid during a terminology migration.

## Internal project layout

For new `barotropic_tide` projects, the [fixed agent workspace contract](barotropic_agent_layout.md) adds the following role roots; `benchmark` projects do not require them:

```text
<project>/agents/00_lead/
<project>/agents/01_grid/
<project>/agents/02_tpxo/
<project>/agents/03_configuration_build/
```

The scientific product layout remains:

```text
<project>/
  gridding/accepted_t6v6/
  gridding/fresh_reproduction/
  forcing/shared_source/
  forcing/accepted_t6v6/
  forcing/fresh_reproduction/
  input/<grid_case>/base/
  input/<grid_case>/attempts/attempt_NNNN/
  input/<grid_case>/final/
  run/build/
  run/<grid_case>/attempts/attempt_NNNN/
  run/<grid_case>/benchmark/
  output/<grid_case>/station/
  output/<grid_case>/condensed/
  analysis/observations/
  analysis/<grid_case>/
  analysis/validation_report.html
  project_manifest.json
  project_status.json
  commands.jsonl
  report.html
```

Project and worker manifests contain relative artifact paths when inside the project, UTC timestamps, SHA-256 for immutable inputs/outputs, code and source lineage, warnings, blocking reasons, and resume tokens. They never contain credentials.

## Paper profile

`profile: paper_barotropic` selects the [paper workflow](paper_barotropic.md).
Run Control resolves the fixed dates, fresh-only grid, physics, stage limits and
production layout into `request.json` before initialization. Explicit conflicts
fail validation; omission of the profile preserves ordinary defaults. No version
suffix is part of the profile name. Record the installed skill commit separately.
Supply `kestrel.project_dir` for the assigned remote root and `source_dir` for its
project-owned source copy. After native environment discovery, record
`kestrel.build_environment` with `os`, `compiler`, `compiler_options`, `modules`
and an RH8 `fallback_reason` when applicable. Complete this preparation before
freezing the request. Authentication/discovery does not launch a model run.

The paper resolver also fixes `kestrel.source_repository` to
`https://github.com/SharknadoBear/FVCOM_MP.git` and `kestrel.source_commit` to
`f8e4b50d4fd51409febcefc763cbc5dca11870e2`. These identify the independently acquired
base source; record permitted build-portability diffs separately. Build finalization
checks the observed revision and its freeze binding. Keep plastic/sediment features
off; using the FVCOM_MP repository does not enable microplastic physics.
