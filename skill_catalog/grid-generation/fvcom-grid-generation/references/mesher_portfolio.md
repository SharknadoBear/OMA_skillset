# Generator-Neutral Mesher Portfolio

Use this research contract when comparing triangulation engines. Keep the
production route unchanged until a candidate passes the complete regional
matrix.

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
| clean-room constrained Delaunay | yes | yes | no | no | production reference and source-lineage control |
| Gmsh algorithms 1/5/6 | yes | yes | yes | yes | topology breadth and algorithm comparison |
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
- Enforce the same 135,000-node preflight threshold and 150,000-node hard cap.
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

The first Lake portfolio audit found that the adaptive one-dimensional target
can be much coarser than the final slope/hydraulic two-dimensional field at the
same shoreline. Treat that as an input-contract failure, not a triangle
algorithm failure.

First resolve the scientific scale shared by the boundary and field. Compute
the bathymetry-supported floor

\[
h_b=
\left\lceil
\frac{3\sqrt{\Delta x_{95}\Delta y_{95}}}{25\ {\rm m}}
\right\rceil 25\ {\rm m},
\]

then select the smallest 25 m multiple \(h_u\ge h_b\) that satisfies the common
135,000-node metric preflight. Assign \(h_u\) to every solid and island target
and use the case manifest's near-OBC target for every open chain. Preserve all
source vertices even when their chords are shorter than \(h_u\); count and
label those unavoidable constraints `geometry_forced_subgrid` instead of
inventing finer boundary nodes or pretending that the physical resolution is
uniform.

The current research implementation uses a deterministic direct fixed point,
not the wet-distance min-plus equations that appeared in the original design:

1. build provisional `fvcom_size_field_v4` \(H_k\) from the case targets;
2. sample \(H_k\) along every immutable source segment;
3. in the portfolio default `sampled_field` mode, set
   \(h_\Gamma(s)=H_k(\Gamma(s))\), apply the closed-chain lower Lipschitz
   envelope with gradation \(g\), and equidistribute boundary metric length
   while retaining every source vertex and lineage record;
4. rebuild \(H_{k+1}\) from that reconciled boundary;
5. restore the exact continuous boundary trace that a cell-centred raster
   cannot represent,
   \[
   \widehat H(x)=\min\left[
   H_{k+1}^{\rm raster}(x),
   \min_i\{h_{\Gamma,i}+g\|x-x_i\|_2\}
   \right],
   \]
   using deterministic trace samples that include every delivered vertex and
   every edge midpoint;
6. repeat from the immutable source boundary until endpoint-and-midpoint
   \(L/\min(h_\Gamma,\widehat H)\), gradation, and factor-two interface gates
   all pass.

This method is recorded as
`authoritative_source_resampling_plus_rebuilt_field_fixed_point` and explicitly
reports `not_wet_distance_min_plus`. It is sufficient for the present raw
bakeoff only when the independent edge audit passes. A tested one-cell
nearest-wet raster halo made the Lake interface worse and is not part of the
contract. The trace extension uses four deterministic samples per local target
spacing and the 16 nearest samples by default; both controls are recorded in
the scientific bundle hash. It is exact at the audited endpoints and
midpoints and releases normally at the same gradation \(g\).

The standalone reconciler keeps a backward-compatible `minimum` mode,
\(h_\Gamma=\min(h_{\rm source},H)\), for workflows whose explicit source target
must never be coarsened. The research portfolio selects `sampled_field` by
default because retaining a finer source target where \(H\) is coarser creates
the boundary/first-ring jump this preflight is designed to reject. Record and
hash the selected mode; changing the case-budget policy does not silently
change it.

The current trace distance is straight Euclidean distance, not wet-domain
geodesic distance. Report possible refinement leakage through barrier land and
islands as an advisory diagnostic; do not describe this trace extension as the
future barrier-aware wet-distance min-plus solver.

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
record, and recompute the node budget. Make the boundary/interior factor-two
diagnostic a hard bundle preflight; do not rank algorithms while it fails.

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

## Current raw-mesher routing default

Use Gmsh Frontal-Delaunay algorithm 6 as the research raw primary. It supports
zero, single, plural, and cyclic OBC contracts and gave the strongest closed-
lake result after the continuous boundary-trace correction: lower node count,
higher \(q_{L3\sigma}\), lower first-ring \(L/h\), maximum valence 8, and fewer
area-change hotspots than Gmsh Delaunay algorithm 5.

This is a routing default, not a declaration that algorithm 6 wins every
metric or has passed production gates. Run algorithm 5 as the first challenger
whenever algorithm 6 fails a hard gate; it retained somewhat better shape-tail
and singly-connected counts in the cyclic Hawaii control. Keep the clean-room
route as the production reference for supported zero/single noncyclic OBC
topologies, and use MeshAdapt algorithm 1 as an explicit diagnostic rather
than the default. Compare failed candidates metric by metric without a
composite winner.

The executable default policy order is:

1. `gmsh_frontal_delaunay_6` — research raw primary;
2. `gmsh_delaunay_5` — first algorithm challenger;
3. `clean_room_raw` — production reference where topology-compatible;
4. `gmsh_meshadapt_1` — robustness diagnostic.

Production promotion remains withheld until one routed workflow passes every
hard gate across the complete six-case topology matrix. The current three-case
no-conditioning control still requires conditioning for thin-angle,
area-change, valence, and/or singly-connected debt.

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

Report the existing p95 and maximum gates separately for boundary edges,
first-ring triangles, the transition band, and the true interior. Do not hide
boundary defects by excluding them.

## Parallel execution

Parallelize independent case/candidate generation because their output
directories and Gmsh sessions are isolated. Parallel repair workers may propose
changes on disjoint logical patches, but only one coordinator may commit
SHA-bound deltas to a global mesh and run the whole-mesh audit. Never cut the
2DM into independently repaired pieces and stitch them across artificial
seams.
