# `simulation_request_v1`

```yaml
schema: simulation_request_v1
mode: barotropic_tide | benchmark
case_id: string
project_dir: optional absolute or workspace-relative path
grid_policy: accepted_then_fresh
period:
  analysis_start: optional UTC timestamp
  analysis_end: optional UTC timestamp
  spinup_days: optional positive number
physics:
  formulation: three_dimensional_barotropic
  temperature_c: optional number
  salinity_psu: optional number
  sigma_layers: optional positive integer
tpxo:
  constituents: all_available
observations:
  provider: noaa_coops
  station_policy: all_available_inside_wet_domain
  current_policy: downward_all_bin_profiles_only
kestrel:
  source_dir: /home/yhuang168/FVCOM_source_agent_test
  account: hindcastra
  partition: auto
resume_from: optional state or attempt token
```

Dates must be UTC and analysis end is exclusive. Defaults for an unspecified Galveston test are analysis `[2025-04-01T00:00:00Z, 2025-05-01T00:00:00Z)`, seven spin-up days, a forcing start of `2025-03-25T00:00:00Z`, and six-minute forcing/station cadence.

## Fixed project layout

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
