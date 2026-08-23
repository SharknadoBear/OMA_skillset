# Generator-Neutral Mesher Portfolio

Use this contract for the operational Gmsh-6 raw route and for explicit
triangulation-engine comparisons. Forward evidence has promoted deterministic
Gmsh Frontal-Delaunay algorithm 6 to the default; other candidates never run
unless named explicitly.

## Separate intent from implementation

Create one immutable input bundle before starting any candidate. The bundle
contains:

- the accepted boundary package and its SHA-256;
- the full bathymetry artifact and its SHA-256;
- one serialized `fvcom_size_field_v4` and its SHA-256;
- the projected CRS, boundary/OBC contract, node budget, and QA-policy hash;
- the exact hard anchors, island loops, OBC chains, and cyclic flags.

Treat that bundle as the scientific mesh intent. A mesher adapter may translate
the intent into its native API, but it may not replace the size function,
redesign the boundary, change the node budget, or relax a QA gate.

Identify the generator and algorithm independently. For example,
`gmsh/meshadapt-1`, `gmsh/delaunay-5`, and
`gmsh/frontal-delaunay-6` are separate candidates even though they share one
library.

Boundary geometry and boundary discretization are different controls. Every
candidate must retain all source vertices, hard anchors, loop topology, and OBC
identity, but an adapter may require native boundary insertions to recover its
constraints. Record the source-vertex count, delivered-boundary count, and
discretization mode for every candidate. Only call a comparison
**algorithm-only** when those policies match. Otherwise label it
**generator-plus-boundary-policy** and use the metrics as capability evidence,
not as proof that one triangle algorithm is intrinsically better.

The adaptive one-dimensional boundary target and the two-dimensional interior
field must also have an explicit compatibility rule. If the boundary package
deliberately controls curve spacing, do not silently drive the one-dimensional
mesh with a smaller raster value and then audit boundary-adjacent elements
against that smaller value. Either use one shared effective target on both
sides of the interface or report a `boundary_interior_size_contract_conflict`
and withhold target-size attribution.

## Capability routing

Do not force every topology through every generator.

| Adapter | Zero OBC | One noncyclic OBC | Multiple OBCs | Cyclic exterior | Research role |
|---|---:|---:|---:|---:|---|
| clean-room constrained Delaunay | yes | yes | no | no | explicit source-lineage research control |
| Gmsh algorithm 6 | yes | yes | yes | yes | operational default |
| Gmsh algorithms 1/5 | yes | yes | yes | yes | explicit challengers and diagnostics |
| external OceanMesh2D | not integrated | not integrated | not integrated | not integrated | GPL method reference unless an isolated adapter is explicitly added |

Record an unsupported pairing as `capability_not_supported`; do not call it a
mesh-quality failure and do not flatten or merge OBC chains to make it run.

## Two-stage comparison

Evaluate each supported candidate twice:

1. `RAW`: native first-order triangular generation with the declared
   deterministic adapter settings and no OMA topology conditioning.
2. `COMMON_CONDITIONED`: the same bounded, generator-neutral conditioner
   applied to the raw 2DM with the canonical size field and boundary metadata.

Never let a candidate use a private cleanup stage and then label the result
`COMMON_CONDITIONED`. Native generator settings such as the selected Gmsh
algorithm and its configured smoothing count remain part of `RAW` provenance.

The common conditioner must preserve the original number, identity, order, and
cyclicity of OBC chains. The portfolio conditioner supports zero, single, and
plural noncyclic SMS nodestrings by fixing the complete topological boundary and
mapping every chain through node lineage. SMS 2DM does not encode cyclicity, so
route a known cyclic case through `RAW` only unless an external cyclicity
manifest is added; record `common_conditioner_capability_not_supported` rather
than guessing.

## Fairness and promotion

- Refuse comparison across different input-bundle hashes or QA-policy hashes.
- Stratify results by boundary-discretization policy; do not rank unmatched
  policies as an algorithm-only bakeoff.
- Use fresh, non-overwriting candidate directories.
- By default, enforce the same 900,000-node preflight threshold and
  1,000,000-node hard cap. Explicit smaller budgets remain valid reproducibility
  overrides.
- Re-audit the delivered mesh after generation. A delivered count above the
  hard cap is a hard failure even when the metric preflight passed.
- Evaluate every delivered 2DM with `fvcom_mesh_quality_v2`, including the
  canonical target-size `L/h` gates.
- Keep structural failures, generator failures, node-budget failures,
  conditioning failures, and scientific-quality failures as distinct
  taxonomies.
- Record candidate wall time and enforce declared resource ceilings. A timeout
  is capability evidence, not a mesh-quality failure.
- Report a metric-by-metric table. Do not form a composite score or declare a
  winner among candidates that fail any hard gate.
- Retain failed artifacts as immutable evidence.

Start with Lake Ontario because all current adapters support its closed,
zero-OBC topology. Continue with Delaware Bay for a single OBC. Use Long Island
Sound next as a capability-routing case: Gmsh may run both exchange gates, while
the current clean-room adapter must report unsupported rather than flattening
them. Advance useful candidates to the remaining three topology cases only
after these controls are reproducible.

## Boundary/field reconciliation before a ranked run

A one-dimensional boundary target can be much coarser or finer than the final
slope/hydraulic two-dimensional field at the same shoreline. Treat that as an
input-contract failure, not a triangle-algorithm failure.

First resolve the scientific scale shared by the boundary and field. Compute
the bathymetry-supported floor

\[
h_b=
\left\lceil
\frac{3\sqrt{\Delta x_{95}\Delta y_{95}}}{25\ {\rm m}}
\right\rceil 25\ {\rm m},
\]

then select the smallest 25 m multiple \(h_u\ge h_b\) that satisfies the common
900,000-node metric preflight. Here \(\Delta x_{95}\) and \(\Delta y_{95}\)
describe bathymetry raster support; 25 m is only the deterministic rounding and
search quantum. The 1,000,000-node cap is a ceiling, not a requested node count:
selection stops at \(h_b\) when that floor already fits. Assign \(h_u\) to every
solid and island target and use the case manifest's near-OBC target for every
open chain. Reject a configuration whose maximum size is below selected
\(h_u\); do not silently clamp the raster floor and invalidate the node-budget
solution. Preserve all source vertices even when their chords are shorter
than \(h_u\); count and label those unavoidable constraints
`geometry_forced_subgrid` instead of inventing finer boundary nodes or
pretending that the physical resolution is uniform.

Make realized source geometry authoritative before the first field build. For
each immutable source chord of length \(L\), derive

\[
h_{\rm geo}=L,
\]

assign each endpoint the minimum of its incident chord targets, and then take
the conservative minimum of \(h_{\rm geo}\) and the case-policy target. This
does not remove, move, or merge a source vertex. It makes the field and first
interior ring respond to a short delivered chord instead of merely diagnosing
the jump after triangulation.

Sub-bathymetry-floor chord targets remain on the one-dimensional boundary
trace; they do not lower the entire two-dimensional raster below its
bathymetry-supported floor. Use bilinear sampling only when all four stencil
cells are active. At a wet/dry interface, choose the highest-weight active
corner with positive interpolation weight instead of assigning one shared
inactive-halo value that could connect two wet banks across land. At a dry-cell
centre, choose the coarsest covered corner so a fine wrong-bank value cannot
override the boundary trace. This raster-interface guard does not claim global
barrier awareness. In the boundary-trace wrapper, use the raster value only
when the query has positive-weight active support. Otherwise make the trace
authoritative and record
`no_active_support_policy=boundary_trace_authoritative`; the coarsest-covered
fallback remains available to raster-only controls and provenance diagnostics,
but it cannot override the trace. The node-budget integral remains restricted
to active wet cells. It combines an active-only neighboring-raster minimum
with adaptive gradation-Lipschitz lower bounds over as many as \(32\times32\)
subcells. The refinement level is chosen from the within-cell release relative
to the local target. This covers off-centre callback refinement without
importing dry-cell targets or charging one fine trace point to a whole coarse
raster cell.

The current research implementation uses a deterministic direct fixed point,
not the wet-distance min-plus equations that appeared in the original design.
Use gradation \(g=0.10\), boundary/field compatibility factor \(1.5\), and at
most eight fixed-point passes:

1. build provisional `fvcom_size_field_v4` \(H_k\) from the case targets;
2. sample \(H_k\) along every immutable source segment;
3. in the portfolio default `sampled_field` mode, set
   \(h_\Gamma(s)=H_k(\Gamma(s))\), apply the closed-chain lower Lipschitz
   envelope with gradation \(g\), and equidistribute boundary metric length
   while retaining every source vertex and lineage record;
4. rebuild \(H_{k+1}\) from that reconciled boundary;
5. restore a sampled approximation to the continuous boundary trace that a
   cell-centred raster cannot represent,
   \[
   T(x)=\min_i\{h_{\Gamma,i}+g\|x-x_i\|_2\},\qquad
   \widehat H(x)=
   \begin{cases}
   \min\!\left(H_{k+1}^{\rm raster}(x),T(x)\right),
      & \text{positive-weight active raster support},\\
   T(x), & \text{otherwise},
   \end{cases}
   \]
   using deterministic trace samples that include every delivered vertex and
   every edge midpoint;
6. repeat from the immutable source boundary until endpoint-and-midpoint
   \(L/\min(h_\Gamma,\widehat H)\), gradation, and factor-1.5 interface gates
   all pass, or reject the common input bundle after the eighth pass.

This method is recorded as
`authoritative_source_geometry_trace_resampling_plus_rebuilt_field_fixed_point`
and explicitly reports `not_wet_distance_min_plus`. It is sufficient for the
present raw bakeoff only when the independent edge audit passes. The
`fvcom_boundary_trace_sampler_v2` extension uses four deterministic samples per
local target spacing, equidistributed by the metric of the linearly
interpolated endpoint targets. It also includes every audited endpoint and
physical midpoint, and fails safely if the deterministic set would exceed five
million points. Its nearest-neighbour search starts with 16 samples and expands
until a global lower bound proves that no unseen sample can lower the answer.
Query locations are processed in deterministic batches of at most 4,096;
this preserves exact values and aggregate counters while bounding the memory
used by the expanding neighbour arrays.
It is therefore exact over the deterministic sample set—not over the
continuous segment—and releases normally at the same gradation \(g\). The
report records expansion counts and the remaining continuous-sampling
overestimate bound. It also records the no-active-support policy so immutable
older v2 artifacts without that key replay their historical `raster_min`
behavior instead of being silently reinterpreted. A fixed 16-neighbour
truncation is not the v2 contract.

After reconciliation, integrate the final size callback at every active wet
raster-cell centre and use that callback-adjusted interior estimate in the
900,000-node preflight. The stored raster estimate is retained for comparison,
but it cannot authorize triangulation when the sampled trace makes the
callback finer.

For open domains, treat the configured OBC transition distance as a minimum.
Compute the distance required by the declared gradation and use the greater of
the requested and required values. Record requested, required, effective, and
available wet distances, including whether the transfer was auto-extended and
whether the full transfer fits within the connected wet domain.

The standalone reconciler keeps a backward-compatible `minimum` mode,
\(h_\Gamma=\min(h_{\rm source},H)\), for workflows whose explicit source target
must never be coarsened. The research portfolio selects `sampled_field` by
default. Under default geometry continuity, \(H\) already includes the
chord-derived trace, so its subfloor source-geometry targets remain
authoritative; disabling geometry continuity is the explicit route that can
allow a coarser sampled field to replace them. Record and hash both controls.

The current trace distance is straight Euclidean distance, not wet-domain
geodesic distance. Record `barrier_aware=false` and possible refinement leakage
through barrier land and islands as an advisory risk. Do not report leakage as
observed or quantified unless a separate barrier audit was run, and do not
describe this trace extension as the future barrier-aware wet-distance
min-plus solver.

The topology-aware target for a later production revision remains the
wet-distance min-plus construction below. Let \(h_0(x)\) be the provisional
wet-domain dynamical field and \(d_\Omega\) wet-domain distance:

\[
T(s)=\inf_{x\in\Omega}\left[h_0(x)+g\,d_\Omega(x,\Gamma(s))\right],
\qquad
h_\Gamma(s)=\min\left[h_{\mathrm{adaptive}}(s),T(s)\right],
\]

\[
H(x)=\min\left[
h_0(x),
\inf_s\left(h_\Gamma(s)+g\,d_\Omega(x,\Gamma(s))\right)
\right].
\]

Do not claim these min-plus equations are implemented until a barrier-aware
wet-domain solver and its fixtures replace the direct fixed point.
Deterministically resample every curve from the accepted \(h_\Gamma\), retain
every source vertex, hard anchor, kind transition, orientation, and lineage
record, and recompute the node budget. Make the boundary/field factor-1.5
compatibility diagnostic a hard bundle preflight; do not rank algorithms while
it fails.

Use two explicit candidate strata:

- `COMMON_LOCKED`: every adapter receives the same reconciled delivered
  boundary and may not change it. An adapter unable to recover that boundary
  without insertion is unsupported in this stratum.
- `NATIVE_SIZE_DRIVEN`: source geometry and reconciled \(h_\Gamma,H\) are
  common, but native insertion is allowed with complete lineage. Label
  clean-room midpoint recovery `RECOVERY_NATIVE` until its insertion locations
  are driven by \(h_\Gamma\), not merely by missing Delaunay constraints.

Compare algorithms only within a stratum. Compare locked versus native as a
boundary-policy delta.

## Operational raw-mesher default

Use Gmsh Frontal-Delaunay algorithm 6 as the operational raw generator. It
supports zero, single, plural, and cyclic OBC contracts. Use first-order
triangles, one thread, random seed 1, eight native smoothing steps, and disable
algorithm switching on failure. A Gmsh-6 failure stops the operational run;
it never activates a different generator.

The executable default candidate list contains only
`gmsh_frontal_delaunay_6`. Name `gmsh_delaunay_5`, `clean_room_raw`, or
`gmsh_meshadapt_1` explicitly for a research comparison. Native algorithm and
smoothing settings remain raw-generator provenance and do not make the
candidate `COMMON_CONDITIONED`.

Audit target size per edge, not only from the longest triangle edge divided by
one centroid sample:

- boundary edge: \(L_e/h_\Gamma(s_e)\);
- interior edge: \(L_e/H(x_e)\), conservatively sampled at both endpoints and
  the midpoint;
- triangle: the maximum of its three edge ratios.

Evaluate boundary/field compatibility pointwise at endpoint A, endpoint B, and
the midpoint, then take the maximum symmetric ratio
\(\max(h_\Gamma/H,H/h_\Gamma)\). Comparing the minimum of one triplet with the
minimum of the other can false-pass anticorrelated targets and is forbidden.

Audit realized boundary-to-bulk continuity independently. For every constraint
edge of length \(L_e\) and every incident first-ring triangle of area \(A_t\),
define \(h_t=\sqrt{4A_t/\sqrt{3}}\) and report
\(\max(L_e/h_t,h_t/L_e)\). Require both the global and per-chain p95 to be at
most `1.5`, and every global and per-chain maximum to be at most `2.0`. This is
a hard gate, not an advisory plot statistic.

Report the existing p95 and maximum gates separately for boundary edges,
first-ring triangles, the transition band, and the true interior. Do not hide
boundary defects by excluding them. Retain the canonical target-size
triangle-edge gates of p95 at most `1.55` and maximum at most `2.0`; they are
separate from the realized boundary/first-ring scale-ratio gate above.

## Parallel execution

Parallelize independent case/candidate generation because their output
directories and Gmsh sessions are isolated. Parallel repair workers may propose
changes on disjoint logical patches, but only one coordinator may commit
SHA-bound deltas to a global mesh and run the whole-mesh audit. Never cut the
2DM into independently repaired pieces and stitch them across artificial
seams.
