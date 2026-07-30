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

This document describes the frozen first Gmsh experiment: algorithm 6 with its
simple Euclidean OBC `Distance`/`Threshold` field. For a fair generator
comparison using the shared production `fvcom_size_field_v4`, run
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
and hashed; they must satisfy the same manifest contract. A check-only mode
should report missing or rejected prerequisites without starting Gmsh.

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

## Node budget and size field

The common hard cap is 150,000 nodes and the preflight threshold is 135,000.
The bathymetry floor is three times the geometric mean of the projected p95
raster-cell dimensions, rounded upward to 25 m. Generate the one-dimensional
boundary first, then choose the smallest 25 m increment of uniform interior
size `h_u` satisfying

```text
N_boundary + integral_Omega[2 / (sqrt(3) * h(x)^2)] dA <= 135000.
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

If a completed mesh exceeds 150,000 nodes, one deterministic rerun may scale
`h_u` by `sqrt(N_actual / 135000)`. A second overflow is `needs_review`.

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

Execution invokes the same read-only readiness gate used by `--check-only`
before starting Gmsh. Model-loop cases recompute exterior overlap and component
counts and bind the upstream land-crossing evidence; the Long Island Sound
case additionally rechecks rank-1 gate selection, shoreline endpoint snapping,
zero land crossing, exterior overlap, feature containment, and one wet
component. Negative fixtures are rejected by path containment. The immutable
manifest snapshot keeps a run reconstructable even if the source manifest is
later edited.

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
