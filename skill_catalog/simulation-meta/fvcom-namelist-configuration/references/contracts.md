# FVCOM namelist contracts

## Inputs

`simulation_request_v1` supplies the science and period:

```yaml
mode: barotropic_tide | benchmark
case_id: string
period:
  analysis_start: UTC timestamp
  analysis_end: UTC timestamp
  spinup_days: 7
physics:
  formulation: three_dimensional_barotropic
  temperature_c: 20
  salinity_psu: 30
  sigma_levels: 10 # legacy sigma_layers also counts levels
```

An artifact-bindings JSON object supplies attempt-specific names:

```json
{
  "input_dir": "../input/accepted_t6v6/final",
  "output_dir": "../output/accepted_t6v6",
  "grid_file": "galveston_grd.dat",
  "depth_file": "galveston_dep.dat",
  "coriolis_file": "galveston_cor.dat",
  "sigma_file": "galveston_sig.dat",
  "sponge_file": "galveston_spg.dat",
  "obc_file": "galveston_obc.dat",
  "elevation_file": "galveston_tides.nc",
  "station_file": "galveston_station.dat",
  "grid_edge_file": "galveston_grid_edge.dat",
  "grid_edge_read_from_file": true,
  "restart_file": "galveston_restart.nc"
}
```

Numerical controls JSON supplies positive `extstep_seconds` and integer `isplit`. They are attempt-specific and must be recorded in the run-control manifest.

Station JSON is either an array or `{"stations": [...]}`. Each station has `station_id`, `longitude`, `latitude`, `cell_id`, and `depth_m`. `cell_id` is one-based and refers to the final FVCOM element order.

## Suitable water-level proxy candidates

With `containing_cell_or_nearest_wet_cell`, the default remains containing cell,
otherwise unrestricted nearest positive-depth centroid. If a geometry review
rejects that nearest cell (for example, across a barrier into another basin),
`prepare_station_mapping.py --proxy-candidates candidates.json` supports a
predeclared set of suitable cells. The owning mapper selects the nearest centroid
within that set using its projected-metre distance and cell-ID tie break. This
option does not infer basin connectivity, choose from agreement statistics, or
authorize remeshing. The reviewer supplies basin/coast/depth evidence before
model-agreement inspection; the code checks the bindings and deterministic choice.

```json
{
  "schema": "fvcom_waterlevel_proxy_candidates_v1",
  "mesh_sha256": "<exact mesh hash>",
  "grid_contract_sha256": "<exact delivery contract hash>",
  "station_inventory_sha256": "<exact source-eligibility inventory hash>",
  "metric_crs": "EPSG:32617",
  "selection_basis": "geometry_only_before_agreement_statistics",
  "model_agreement_used": false,
  "stations": [{
    "station_id": "example_station",
    "observation_longitude": -81.0,
    "observation_latitude": 27.0,
    "candidate_cell_ids": [101, 102],
    "basin_rationale": "Reviewed connected harbor cells; offshore cells across the barrier excluded.",
    "geometry_evidence": [{"path": "basin_review.json", "sha256": "<exact evidence hash>"}]
  }]
}
```

Evidence paths may be absolute or relative to the candidate manifest. Every file
must exist and match its hash. Candidate IDs must be unique integers referring
to finite positive-mean-depth cells in the bound mesh. Gauge coordinates must
exactly match the inventory. Constraints apply only to outside water-level
gauges with eligible source data (or whose sole exclusion is wet containment);
currents and containing-cell selections cannot be overridden.

The new mapping method is `nearest_suitable_wet_cell_centroid`. It retains the
original/model coordinates, depth, node IDs, equal three-node weights, distance,
truthful containment and the complete `unrestricted_nearest` geometry. It also
records `candidate_set_sha256`, candidate count, rationale and evidence bindings.
The top-level mapping records `proxy_candidates_sha256`. An ordinary passed
`fvcom_waterlevel_proxy_review_v1` still approves the exact final geometry; each
constrained station's review record must additionally contain the matching
`candidate_set_sha256`. Changed candidate lists or rationale invalidate that
approval even if the selected cell happens to stay the same.

Use `candidate_cell_ids: []` with the same evidence requirements when no suitable
cell is available. The gauge is explicitly excluded as
`no_suitable_wet_cell_candidate`, retaining the rejected nearest candidate and
rationale. If no comparison stations remain, the mapper publishes a `blocked`
mapping with that disposition, not a ready empty validation set. Other viable
gauges and existing diagnostic rows remain separate. None of these decisions
establishes scientific representativeness or alters source-data eligibility.

## Stage timing

- `smoke`: cold start, one hour from spin-up start.
- `canary`: cold start, three days from spin-up start, covering the full two-day ramp.
- `spinup`: cold start, seven days ending at `analysis_start`; restart output begins exactly at `analysis_start`.
- `benchmark`: hot start at `analysis_start`, one simulated day.
- `production`: hot start at `analysis_start`, ending at `analysis_end`.
- `full`: cold start from spin-up start through `analysis_end`.

## Duration tokens

FVCOM 4.3.1 `IDEAL_TIME_STRING2TIME` accepts lowercase `seconds`, `days`,
or `cycles`. Its `GET_VALUE` helper classifies a number as real only when
the string contains a decimal point. Use `seconds=36.0`, `seconds=3.6e1`,
or `days=1.0`; use integer tokens for cycles, such as `cycles=10`.
E/e exponents with a decimal point are supported; D/d exponents and
`hours`/`minutes` units are not. A Python numeric conversion alone cannot
certify this syntax. Audit active interval fields before submission;
inactive output placeholders are not executable durations. This lexical
gate covers one scalar token, not GET_VALUE's list/comment syntax or all
Fortran conversion limits. It does not replace numeric-range, positive-interval,
stage-clock and internal-step divisibility checks.

## Template lineage

The final namelist is an overlay on `blank_namelist.nml` generated by the frozen FVCOM executable. Record both hashes. If a required group or key is absent, stop: the executable and configuration contract do not match.

The paper workflow uses the pinned FVCOM_MP source family. Its `FLUME_BOTTOM`,
`FLUME_DRAG_COEF`, `READ_GRID_EDGE_FROM_FILE`, and `GRID_EDGE_FILE_NAME` fields are
custom extensions absent from upstream release-v4.3.1. The paper overlay also
explicitly disables `PLASTIC_MODEL`; this is a hydrodynamic tide-only run.

FVCOM 4.3.1 input definitions are authoritative for field spelling. Important groups are `NML_CASE`, `NML_STARTUP`, `NML_IO`, `NML_INTEGRATION`, `NML_RESTART`, `NML_NETCDF`, `NML_PHYSICS`, `NML_SURFACE_FORCING`, `NML_RIVER_TYPE`, `NML_OPEN_BOUNDARY_CONTROL`, `NML_GRID_COORDINATES`, `NML_GROUNDWATER`, `NML_ADDITIONAL_MODELS`, `NML_PROBES`, and `NML_STATION_TIMESERIES`.

## File gate

A final configuration requires readable grid, depth, Coriolis, sigma, sponge, OBC-node, elevation-forcing, and station files. Set `grid_edge_read_from_file=false` only for the first cold smoke that will generate the named adjacency artifact into its attempt workspace. After that stable smoke, freeze the generated file and set the binding to `true`; subsequent cold canary/spin-up stages and all hot-start benchmark/production stages then require and reuse it. Hot starts with the binding set to false are rejected. The preconfiguration manifest must report ten sigma levels (nine layers) and hashes consistent with the selected files.

The configuration manifest records the namelist/station/template hashes, stage, timing, numerics, all file checks, warnings, blocking reasons, and a resume token.

## paper_barotropic override

`profile: paper_barotropic` resolves the fixed common settings from Run Control.
Only smoke (March 29–30, 2025 UTC) and production (March 29–April 15 UTC) exist;
both cold-start and emit a final restart. Analysis is April 1–15, end exclusive.
There is no hotstart prerequisite or separate canary/spin-up/benchmark stage.
All other timing, artifact-binding and tide-only checks remain in force.
