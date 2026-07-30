# Research Gmsh FVCOM grid experiment

Status: **PROPOSED / RESEARCH ONLY**

This directory contains an isolated Gmsh alternative for the six-case FVCOM
grid experiment. It is not a production backend, and its presence must not
change the default `fvcom-grid-generation` workflow. Production promotion is
out of scope until all six topology cases pass every structural and quality
gate.

The case manifests point to immutable workspace inputs. They declare input
readiness, not mesh acceptance. Every run must use a new output directory.
The completed six-case baseline remains research evidence only: all six
regional meshes are classified `needs_review`, so production promotion is
ineligible.

The archived baseline exercised algorithm 6 with its simple Euclidean OBC
`Distance`/`Threshold` field. Its immutable run snapshots retain the original
135,000-node preflight threshold and 150,000-node hard cap. Current source
manifests and command defaults use the larger future-run policy documented
below. For a fair generator comparison using the shared production
`fvcom_size_field_v4`, run
`scripts/run_mesher_portfolio_case.py`; that portfolio evaluates Gmsh
algorithms 1, 5, and 6 plus the clean-room reference from one hashed scientific
input bundle. Do not mix the two size-field contracts in one comparison.

## Environment

Use 64-bit CPython 3.13 in a project-local virtual environment. If CPython is
not present, install it per-user with `PrependPath=0`; do not change the global
`PATH`. Invoke the interpreter by its explicit path:

```powershell
$pythonExe = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
& $pythonExe -m venv .venv-gmsh-4.15.2
& .\.venv-gmsh-4.15.2\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-gmsh-4.15.2\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins the official Python distribution to `gmsh==4.15.2`.
The supporting scientific packages use bounded ranges so a Python 3.13 wheel
can be resolved. Each run manifest must record the exact Python, Gmsh,
platform, and resolved dependency versions, plus SHA-256 hashes for every
input.

Do not vendor Gmsh source or binaries into this repository. Gmsh is distributed
under the GNU General Public License. Internal execution is distinct from
redistribution; any redistribution of Gmsh, its wheel, a combined binary, or a
package that includes it must be reviewed for the applicable GPL obligations,
including corresponding-source and license-notice requirements.

## Case manifests and readiness

All paths in `cases/*.json` are relative to the workspace root
(`OMA_intiate`). The preparation phase brought all six current manifests to
input-ready status:

| Order | Case | Boundary | Bathymetry | Manifest status |
|---:|---|---|---|---|
| 1 | San Francisco Bay | corrected adaptive-v2 available | available | `ready` |
| 2 | Delaware Bay | prevention-only adaptive-v2 available | frozen full coverage available | `ready` |
| 3 | Galveston–Trinity Bay | corrected `g1` revalidated; stale `gtb` rejected | CUDEM/BlueTopo/ETOPO complete | `ready` |
| 4 | Long Island Sound | automatic two-gate package rebuilt | full wet footprint complete | `ready` |
| 5 | Lake Ontario | closed GSHHG L1/L2/L3 package built | full-lake ETOPO complete | `ready` |
| 6 | Hawaiian Islands | `hi` island-loop geometry revalidated | CRM/ETOPO complete | `ready` |

The stale Galveston `gtb` package and the land-crossing Long Island Sound
package are negative rejection fixtures only. A readiness check must reject
them; it must never silently substitute either for the active input.

## Lake Superior continuity-case preparation

`continuity_cases/lake_superior.json` is the authority for the separate
three-case boundary-continuity experiment; it is not a seventh member of the
archived six-case matrix. The current accepted selection is exactly:

- boundary v5:
  `Workspace/Preprocessing/fvcom-grid-generation/runs/cont2_lsup_prep_20260730/03_lake_superior/prep_boundary_v5/boundary_resolution_manifest.json`;
- bathymetry v2:
  `Workspace/Preprocessing/fvcom-grid-generation/runs/cont2_lsup_prep_20260730/03_lake_superior/prep_bathymetry/lake_superior_fvcom_depth_v2.nc`.

The unversioned `prep_boundary` and `prep_boundary_v2` through
`prep_boundary_v4` directories, the unversioned
`lake_superior_fvcom_depth.nc`, and readiness reports that bind those inputs
are superseded immutable evidence. Do not select them by filename order or use
them as automatic fallbacks. The currently selected readiness evidence is
`lake_superior_preparation_readiness_v2.json`; despite its suffix, it is the
report that binds boundary v5 and bathymetry v2.

Use the following estimate-first pattern from the workspace root. Replace
`<fresh-rebuild-id>` once with a new run identifier; none of the output
directories may already contain artifacts.

```powershell
$workspace = (Resolve-Path ".").Path
$skill = Join-Path $workspace "Agent_skill_dev/skill_catalog/grid-generation/fvcom-grid-generation"
$cudem = Join-Path $workspace "Agent_skill_dev/skill_catalog/external-data-connectors/cudem-bathy"
$python = Join-Path $workspace "Agent_skill_dev/.venv-gmsh-4.15.2/Scripts/python.exe"
$fresh = Join-Path $workspace "Workspace/Preprocessing/fvcom-grid-generation/runs/<fresh-rebuild-id>/03_lake_superior"
$boundaryEstimate = Join-Path $fresh "prep_boundary_estimate"
$boundary = Join-Path $fresh "prep_boundary"
$bathy = Join-Path $fresh "prep_bathymetry"
$requestTemplate = Join-Path $skill "scripts/research/gmsh/continuity_cases/lake_superior_etopo_request.json"
$sourceIndex = Join-Path $bathy "bathy_source_index_etopo.json"
```

Estimate the GSHHG request, inspect `download_estimate.json`, and only then
prepare the boundary. Omit `--allow-download` when the frozen cache is
complete; add it only after the estimate gate when the official cache must be
filled.

```powershell
& $python (Join-Path $skill "scripts/research/gmsh/prepare_lake_superior_boundary.py") `
  --workspace-root $workspace `
  --output-dir $boundaryEstimate `
  --estimate-only

& $python (Join-Path $skill "scripts/research/gmsh/prepare_lake_superior_boundary.py") `
  --workspace-root $workspace `
  --output-dir $boundary
```

Create a fresh request-bounded ETOPO work directory, run the `cudem-bathy`
estimate before any live fetch, follow its small-smoke-test requirement for a
new transport environment, and then fetch the full 15-arcsecond footprint.
Proceed locally only when the estimate confirms free space greater than four
times the requested bytes; otherwise use the connector's documented Kestrel
route. The committed request template pins bbox
`[-92.2, 46.25, -84.0, 49.1]`, ETOPO-only source order, 15-arcsecond spacing,
the two intersecting official ETOPO tile URLs, and the conservative 128 MiB
estimate. Build a fresh ETOPO-only source index from the live official catalog;
no dated workspace request or source-index file is a prerequisite.

```powershell
New-Item -ItemType Directory -Path $bathy | Out-Null
Copy-Item -LiteralPath $requestTemplate -Destination (Join-Path $bathy "request.json")

& $python (Join-Path $cudem "scripts/estimate_data_request.py") `
  --request (Join-Path $bathy "request.json") `
  --run-dir $bathy `
  --output (Join-Path $bathy "download_estimate.json")

& $python (Join-Path $cudem "scripts/build_bathy_source_index.py") `
  --output $sourceIndex `
  --no-cudem `
  --no-nbs `
  --no-crm

& $python (Join-Path $cudem "scripts/fetch_bathy_sources.py") `
  --bbox -92.2 46.25 -84.0 49.1 `
  --run-dir $bathy `
  --name lake_superior_etopo `
  --index $sourceIndex `
  --fallback-policy cudem-crm-etopo `
  --resolution-policy source-priority `
  --target-spacing-arcsec 15
```

Convert the immutable ETOPO elevation mosaic to positive-down depth using the
fresh accepted boundary mask, then run the connector health gate.

```powershell
& $python (Join-Path $skill "scripts/research/gmsh/prepare_lake_superior_bathymetry.py") `
  --input (Join-Path $bathy "lake_superior_etopo_bathy_sources.nc") `
  --output (Join-Path $bathy "lake_superior_fvcom_depth.nc") `
  --metadata (Join-Path $bathy "lake_superior_depth_conversion.json") `
  --domain-gpkg (Join-Path $boundary "boundary_resolution.gpkg")

& $python (Join-Path $cudem "scripts/check_download_health.py") `
  --request (Join-Path $bathy "request.json") `
  --run-dir $bathy `
  --output (Join-Path $bathy "health_check.json") `
  --plots-dir (Join-Path $bathy "health_plots")
```

For a rebuilt preparation, copy `continuity_cases/lake_superior.json` to a
fresh draft manifest and change `boundary.resolution_manifest` and
`bathymetry.netcdf` to the fresh outputs. Keep the checked-in selector on v5/v2
until the replacement has been reviewed. The validator must write a second,
fresh bound manifest whose `readiness.validation_artifact` path, digest,
required checks, and active-input hashes point to the new evidence. Changing
only the two input paths is not runnable because it leaves the old readiness
binding stale. The runtime also rejects every explicit input override for Lake
Superior; select changed inputs only through a freshly validated bound case
manifest.

```powershell
$caseDraft = Join-Path $fresh "lake_superior.draft.json"
$case = Join-Path $fresh "lake_superior.bound.json"
$readiness = Join-Path $fresh "selected_preparation_readiness.json"
$raw = Join-Path $fresh "raw_portfolio"

# Copy the checked-in selector to $caseDraft, then edit only the two fresh
# boundary and bathymetry input paths described above.
& $python (Join-Path $skill "scripts/research/gmsh/validate_lake_superior_preparation.py") `
  --workspace-root $workspace `
  --case-manifest $caseDraft `
  --output $readiness `
  --bound-case-manifest-output $case

& $python (Join-Path $skill "scripts/run_gmsh_fvcom.py") `
  --workspace-root $workspace `
  --case-manifest $case `
  --check-only

& $python (Join-Path $skill "scripts/run_mesher_portfolio_case.py") `
  --workspace-root $workspace `
  --case-manifest $case `
  --output-dir $raw `
  --candidates gmsh-6 gmsh-5
```

Require validator `status=ready` and a passing check-only audit before either
raw candidate. To run the currently accepted v5/v2 selection, use the checked-
in bound manifest directly:

```powershell
$case = Join-Path $skill "scripts/research/gmsh/continuity_cases/lake_superior.json"
$raw = Join-Path $fresh "raw_portfolio"

& $python (Join-Path $skill "scripts/run_gmsh_fvcom.py") `
  --workspace-root $workspace `
  --case-manifest $case `
  --check-only

& $python (Join-Path $skill "scripts/run_mesher_portfolio_case.py") `
  --workspace-root $workspace `
  --case-manifest $case `
  --output-dir $raw `
  --candidates gmsh-6 gmsh-5
```

The validator and portfolio outputs are new immutable evidence. Never rerun
into an existing preparation or candidate directory, never mutate a prior
manifest to point at newer files, and never start raw generation after a
non-`ready` validation. The portfolio command is raw-only and applies no OMA
conditioning or postprocessing.

## Research CLI contract

The research entry point accepts a case manifest and allows explicit input
overrides:

```powershell
& .\.venv-gmsh-4.15.2\Scripts\python.exe run_gmsh_fvcom.py `
  --case-manifest cases/01_san_francisco_bay.json `
  --output-dir <fresh-run-directory>
```

The interface also accepts `--boundary-loop-package`,
`--adaptive-resolution-manifest`, `--bathymetry-netcdf`,
`--preflight-node-threshold`, and `--hard-node-cap`. Overrides must be recorded
and hashed for legacy research cases. Lake Superior rejects all overrides
because its readiness artifact binds the active paths and hashes; use a freshly
validated bound manifest instead. A check-only mode reports missing or rejected
prerequisites without starting Gmsh.

## Shared-field raw continuity portfolio

Use `scripts/run_mesher_portfolio_case.py` for the current boundary-to-bulk
continuity experiment. It is distinct from the frozen direct six-case
`Distance`/`Threshold` experiment below. The portfolio hashes one canonical
`fvcom_size_field_v4` input bundle and uses Gmsh Frontal-Delaunay algorithm 6 as
the raw primary, with Gmsh Delaunay algorithm 5 as the first hard-gate
challenger. This is routing policy only; it is not a regional-result,
algorithm-winner, or production-promotion claim.

The portfolio defaults are a 900,000-node planning threshold, a 1,000,000-node
hard cap, gradation `0.10`, boundary/field compatibility factor `1.5`, and at
most eight boundary/field fixed-point passes. For every immutable source chord
of length \(L\), derive the geometry-aware endpoint target
\(h_{\rm geo}=L\), take the conservative minimum with the case-policy
target, and preserve every original vertex, chain transition, anchor,
orientation, and lineage record.

Keep sub-bathymetry-floor chord targets on the one-dimensional boundary trace;
the two-dimensional raster retains the bathymetry-supported interior floor.
Use bilinear sampling only for an all-active stencil. At a wet/dry interface,
select the highest-weight positive-weight active corner instead of sharing an
inactive halo between wet banks. At an exact dry-cell centre, use the coarsest
covered fallback so a fine target cannot leak from the wrong bank. This
raster-interface guard is not a global wet-path solver. When the trace wrapper
has no positive-weight active raster support, make the trace authoritative and
record `no_active_support_policy=boundary_trace_authoritative`; retain the
coarsest-covered value only for raster-only controls and provenance replay.
Keep the node-budget integral restricted to active wet cells. Preflight
combines an active-only neighboring-raster minimum with the trace's
adaptive gradation-Lipschitz lower bounds over up to \(32\times32\) subcells;
dry-cell values cannot drive that bound, and an isolated fine trace point is
not charged to a complete coarse raster cell.

Reconstruct a sampled approximation to the continuous boundary extension with
`fvcom_boundary_trace_sampler_v2`. Equidistribute samples by the metric of the
linearly interpolated endpoint targets, include every delivered endpoint and
physical midpoint, and fail safely above five million samples. A query begins
with 16 nearest samples and expands until a global lower bound proves the exact
minimum over that deterministic set; this is not a claim of exact analytic
minimization over each continuous segment. Batch query locations in groups of
at most 4,096 to bound adaptive-neighbour memory without changing values or
aggregate counters. Because this callback can be finer than the stored
cell-centred raster, recompute the interior node-budget
estimate with active neighboring-raster support and adaptive trace subcells
before applying the 900,000-node preflight. Each trace subcell subtracts its
maximum Lipschitz release from the exact subcell-centre trace value, capped by
the global minimum trace target.

Raw diagnostics dispatch the recorded raster-interface schema: archived
`fvcom_wet_mask_sampling_v1` and current v2 semantics are replayed separately.
For trace-sampler v2 artifacts, the recorded `no_active_support_policy` is also
authoritative; an older artifact without that key replays historical
`raster_min` behavior.

For an open domain, use the configured OBC transfer distance as a minimum and
extend it automatically when the `0.10` gradation requires a longer connected-
wet-domain transfer. Record requested, required, effective, and available
distances. After triangulation, compare every realized constraint-edge length
with the equal-area equilateral characteristic length of each incident
first-ring triangle. Require the symmetric ratio to have p95 at most `1.5` and
maximum at most `2.0`, globally and for every chain. This hard continuity gate
is separate from the canonical target-size `L/h` gates.

These comparisons stop at `RAW`. Disable all OMA conditioning and
postprocessing. Native Gmsh algorithm and smoothing settings remain raw
generator provenance and must not be labelled `COMMON_CONDITIONED`.

## Geometry and boundary contract

- Use the built-in `geo` kernel. Every source vertex is a CAD point and every
  source segment is a CAD line.
- The exterior is the first curve loop. Island boundaries are hole loops.
- Create physical groups for `WET_DOMAIN`, every OBC chain, land coastline,
  and every island.
- Gmsh may insert nodes along a source segment, but it may not remove an
  original vertex, hard landfall, or chain-kind transition.
- Record source chain, source segment, interpolation weight, and normalized
  arclength for every delivered boundary node.
- Any changed OBC node sequence sets `forcing_compatible=false`. This
  experiment does not interpolate existing forcing.
- Write one independently terminated SMS nodestring per OBC chain. Long
  nodestrings may span multiple physical `NS` lines. A cyclic chain omits a
  repeated first node, and the last-to-first mesh edge is audited explicitly.

Plural, zero, and cyclic OBCs are end-to-end contracts for this research Gmsh
route. The existing production clean-room mesher still supports at most one
noncyclic OBC and fails fast on explicit multiple or cyclic contract shapes;
it does not silently flatten them.

## Direct six-case node budget and size field

The current common hard cap is 1,000,000 nodes and the planning threshold is
900,000, preserving ten percent headroom. These are defaults for fresh runs;
explicit smaller values remain available for baseline reproduction. The
bathymetry floor is three times the geometric mean of the projected p95
raster-cell dimensions, rounded upward on a 25 m numerical grid. The projected
p95 dimensions are the bathymetry resolution evidence; 25 m is only the
rounding and `h_u` search quantum. Generate the one-dimensional boundary first,
then choose the smallest 25 m increment of uniform interior size `h_u`
satisfying

```text
N_boundary + integral_Omega[2 / (sqrt(3) * h(x)^2)] dA <= 900000.
```

Open-boundary cases use one Gmsh `Distance` field over all OBC curves and a
linear `Threshold`: 8 km elements through 10 km from the OBC, transitioning
to `h_u` at 70 km. Lake Ontario uses a constant `h_u` field. Point, curvature,
and boundary-extension sizing are disabled. A synthetic fixture must verify
the intentionally reversed numerical `SizeMin` and `SizeMax` values at
0/10/70 km before regional runs.

The mesh is first-order triangles with algorithm 6 (Frontal-Delaunay), eight
native smoothing steps, no recombination, no Netgen optimization, and no OMA
conditioning or postprocessing. Use one thread, a fixed random seed,
reproducible mode, and disabled automatic algorithm switching.

If a completed mesh exceeds 1,000,000 nodes, one deterministic rerun may scale
`h_u` by `sqrt(N_actual / 900000)`. A second overflow is `needs_review`.

## Required artifacts

Each fresh run directory contains, at minimum:

- canonical MSH 4.1;
- FVCOM/SMS 2DM;
- an exact case-manifest snapshot, automatic readiness report, and boundary
  revalidation report;
- run and environment manifests with input hashes;
- node-budget preflight;
- sampled source-aware size field;
- boundary-remap/lineage manifest;
- existing-formula quality JSON;
- domain, mesh, quality, and size-field maps;
- Gmsh logger output.

Automatic manifest-driven execution invokes the same read-only readiness gate
used by `--check-only` before starting Gmsh, then proves the prepared manifest
hash matches the readiness hash. Legacy direct runs with explicit overrides
record that exception and rerun coverage checks; Lake Superior cannot use that
route. Model-loop cases recompute exterior overlap and component counts and bind
the upstream land-crossing evidence; the Long Island Sound case additionally
rechecks rank-1 gate selection, shoreline endpoint snapping, zero land crossing,
exterior overlap, feature containment, and one wet component. Negative fixtures
are rejected by path containment. The immutable manifest snapshot keeps a run
reconstructable even if the source manifest is later edited.

The quality report includes the hard FVCOM gates plus advisory Gmsh
SICN/gamma, OBC normality, and Euclidean-through-land sizing leakage. Archived
San Francisco and Delaware meshes are contextual comparisons only; there is no
composite winner and no new in-house rerun is required.

## Known first-experiment risk

Gmsh `Distance` is straight Euclidean distance. It can transmit coarse OBC
sizing across barrier land or islands. The experiment must measure and report
that leakage, but must not correct it in this first route. Other hard-stop
conditions include stale or failed upstream boundary packages, missing
full-footprint bathymetry, invalid OBC order, topology loss, a second node-cap
overflow, nondeterministic repeat output, or any existing FVCOM quality-gate
failure.

## Lake Ontario depth interpretation

`prepare_lake_ontario_bathymetry.py` retains the ETOPO bed elevation and
derives FVCOM positive-down depth from the NOAA 74.2 m IGLD 1985 chart datum.
The output records that ETOPO's EGM2008 vertical reference is not transformed
into IGLD 1985; this is a research caveat, not silent datum harmonization.
Datum reference: <https://tidesandcurrents.noaa.gov/gldatums.html>.

After all six immutable regional runs exist, `summarize_six_case_runs.py`
collects their run manifests, preflight budgets, quality reports, advisory
diagnostics, map paths, boundary-revalidation results, and exact
case-manifest-snapshot hashes into a canonical JSON/CSV matrix. It refuses
missing or duplicate case runs and reports production eligibility only when
all six revalidations and all six hard-gate statuses pass.
