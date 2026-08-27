---
name: fvcom-grid-generation
description: Invoke fvcom-region-bpoly, fvcom-bdry-arc, and cudem-bathy as returning subworkflows for a regional scientific request, then generate benchmark-ready SMS 2DM meshes with mandatory Adaptive v2 boundaries, deterministic Gmsh Frontal-Delaunay algorithm 6, exact per-OBC chains, geometry-derived sizing, topology-priority conditioning, standardized project delivery, and FVCOM QA.
---

# fvcom-grid-generation

## Returning Upstream Subworkflows

For a new regional scientific request, this skill owns the complete upstream
orchestration. Invoke these skills in order and wait for each return:

```text
request -> S1 fvcom-region-bpoly -> S2 fvcom-bdry-arc -> S3 cudem-bathy -> grid generation
```

1. Invoke `$fvcom-region-bpoly` as returning subworkflow **S1** with the
   original scientific request. Receive its canonical `region_bpoly.json`,
   `offshore_boundary_artifacts.json`, and manifest without changing the
   returned geometry.
2. Invoke `$fvcom-bdry-arc` as returning subworkflow **S2** with the original
   request and the exact same-run S1 return. S2 must consume that return
   directly and must not invoke RegionBPoly a second time. Receive the
   finalized boundary-arc manifest, continuous model loops, active v2
   open-exterior evidence, and passing `adaptive-coastal-v2` resolution
   manifest.
3. After S2 defines the finalized assembled wet domain, invoke `$cudem-bathy`
   as returning subworkflow **S3**. Derive its request from that domain plus
   the configured projected halo, and receive the source manifest, fetched or
   mosaicked bathymetry, health evidence, and provenance.
4. Resume this parent skill with the three returned packages. This skill owns
   delivered-OBC bathymetry support, mesh intent, Gmsh generation,
   conditioning, QA, and publication.

Do not call S3 from the earlier RegionBPoly envelope. On a same-run resume,
reuse a complete returned stage rather than invoking it twice. The explicit
artifact interface remains a compatibility branch: when complete boundary,
v2 resolution, and bathymetry artifacts are supplied, validate and consume
them directly; when only boundary artifacts are supplied, invoke only S3 for
the missing bathymetry. Record these branches as supplied-artifact reuse, not
as subworkflow calls that did not occur.

## Core Rules

- Reuse upstream domain and boundary artifacts; do not redesign the region here.
- Require a passing `adaptive-coastal-v2` boundary-resolution manifest for every new grid. Consume its exact per-`obc_id` node sequences without concatenation. Archived legacy/v1 packages remain readable only for provenance and inspection; never use them to generate a new mesh.
- Use deterministic Gmsh Frontal-Delaunay algorithm 6 for every new operational/full-project raw mesh: one thread, seed 1, first-order triangles, eight native smoothing steps, and no algorithm fallback. The default portfolio executes only this candidate. Keep Gmsh 1/5 and clean-room SciPy-Delaunay available only when explicitly requested as research controls; never silently substitute them after a Gmsh-6 failure.
- Revalidate the active `fvcom_open_exterior_contract_v2` evidence at every coastal entry point. Treat approved lagoon closures as fixed solid boundary, require station evidence and requested-count permission for secondary tidal OBCs, and reject unassigned water, missing or stale decisions/maps, invalid physical landfalls or topology, report-only packages, or `downstream_eligible=false` even when an upstream manifest says `pass`. Archived v1 is inspection-only and historical v3 is unsupported.
- Revalidate `fvcom_coastline_source_coverage_v1` with every coastal open-exterior contract. Require at least 2x centered coverage, a physical-coastline-only landfall lineage, zero source-frame dependency, and current whole/zoom map hashes. Historical exact-bbox evidence remains readable but is ineligible for new standardized projects without this proof.
- Keep full bathymetry for final node sampling and bound only the in-memory size-field grid.
- For automatic acquisition, derive the bathymetry request bbox from the
  assembled `model_domain_polygon` with the configured projected halo (2 km by
  default), never from the earlier RegionBPoly envelope. Require at least 95%
  finite sampled wet-domain coverage and 100% finite support along every
  delivered OBC before size-field construction. Sample the OBC at no more than
  half the minimum raster-axis step and use `NaN`, not a median fallback,
  outside source support.
- Build nearshore sizing from the solid-boundary background, bathymetric gradient, and a geometry-derived paired-bank hydraulic skeleton. Detect the skeleton from land and island segments only; never use an open-boundary segment as a bank.
- Do not consume or generate an external drainage/flow-network package. Infer hydraulic corridors directly from the wet model polygon, solid boundary geometry, and the supplied bathymetry.
- For open domains, propagate the delivered OBC target and distance through connected wet cells, hold offshore authority near the OBC, then transfer smoothly in log space to the nearshore target. Treat the configured transfer distance as a minimum and extend it automatically when the declared gradation requires more wet distance. For closed domains, omit this transfer.
- Apply all distance propagation and final gradation over the wet model domain so land and island holes cannot create shortcuts.
- Treat boundary, bathymetry, `fvcom_size_field_v4`, node budget, and QA policy as generator-neutral mesh intent. Lock their hashes before a multi-mesher test, choose adapters by topology capability, and never flatten OBC chains to make an unsupported generator run.
- Load and obey [the benchmark-first quality policy](references/fvcom_grid_quality_policy_v1.json) for every evaluation, conditioning decision, portfolio result, and standardized publication. Fail closed when its hash is stale or a finding has no unique bucket.
- Resolve `--conditioning-profile auto` to `minimal-topology-v1`. Run only fixed-boundary valence repair, immediate superthin cleanup, residual protected-edge-safe superthin repair, and a terminal valence/thin scan. Preserve absolute structural invariants; prioritize valence debt, then superthin debt; never roll either repair back because of Class-2 angle, quality, area-transition, slope, singly-connected-count, or size-continuity changes. Do not run spring relaxation, area-transition relaxation, pruning, boundary edits, global retriangulation, or the broad legacy postprocessor implicitly.
- Keep `autonomous-thin-v1` opt-in. After minimal conditioning, have Codex inspect hash-bound whole-domain and component diagrams, classify the causal mechanism, and route it to bounded local repair, resolution refinement, localized CUSP/GSHHS boundary regularization, or complete subgrid-connection closure followed by full Gmsh-6 remeshing. Do not change `auto` until forward testing is reviewed, and never insert a routine human-review gate or delete one triangle in isolation. Read `references/autonomous_thin_boundary.md` before using this profile.
- Keep depths finite and positive down and treat OceanMesh2D GPL material as a method reference only.

## Primary Workflow

For every new complete grid, initialize the standardized portable project first. Keep attempts under each stage's `_work/`, promote one hash-bound selection to its canonical stage name, and publish any terminal mesh at the stable path `final/fvcom_grid.2dm`. The filename does not imply readiness. Use `validate --require-benchmark-ready` before a first benchmark and `validate --require-submission-ready` before submission.

```powershell
python scripts/manage_fvcom_grid_project.py init --project runs/my_project --name my_project
python scripts/run_mesher_portfolio_case.py --case-manifest runs/my_project/05_mesh_intent/case_manifest.json --output-dir runs/my_project/06_raw_mesh/_work/gmsh6
python scripts/manage_fvcom_grid_project.py promote --project runs/my_project --stage 06_raw_mesh --source runs/my_project/06_raw_mesh/_work/gmsh6/candidates/gmsh_frontal_delaunay_6/raw_mesh.2dm --artifact-name raw_mesh.2dm --generator-manifest runs/my_project/06_raw_mesh/_work/gmsh6/candidates/gmsh_frontal_delaunay_6/candidate_manifest.json
python scripts/manage_fvcom_grid_project.py validate --project runs/my_project
```

See `references/grid_project_contract.md` for the fixed layout, canonical artifacts, publication inputs, and submission gate. Existing lower-level `--run-dir` commands remain supported for historical evidence.

Adaptive package:

```powershell
python scripts/run_fvcom_grid.py --bdry-arc-manifest bdry_arc_manifest.json --boundary-loops-gpkg model_boundary_loops.gpkg --boundary-resolution-manifest boundary_resolution_manifest.json --bathy-nc bathy.nc --run-dir runs/case --name case --mode test --postprocess-profile none
```

Use this command only for test-mode mesh-intent/smoke evidence. Full clean-room
execution requires the explicit `--allow-clean-room-execute` research override
and is ineligible for standardized operational publication. A new grid without
a passing v2 resolution manifest is rejected.

## Research Mesher Portfolio

Use Gmsh Frontal-Delaunay algorithm 6 as the operational raw generator. Keep
the clean-room constrained-Delaunay route as an explicit source-lineage
control and use the other isolated Gmsh adapters only for named research
candidates:
MeshAdapt algorithm 1, Delaunay algorithm 5, and Frontal-Delaunay algorithm 6.
Feed every candidate the same immutable boundary, bathymetry,
`fvcom_size_field_v4`, node budget, and QA policy. Separate the `RAW` generator
result from `COMMON_CONDITIONED`; do not credit generator-specific cleanup as
common conditioning.
By default, derive the bathymetry floor from three times the geometric mean of
the projected p95 raster-cell dimensions, round it upward on a 25 m numerical
grid, and select the smallest budget-compatible uniform target `h_u` on that
same grid. The 25 m value is a deterministic rounding/search quantum, not the
bathymetry resolution. Use a 900,000-node planning threshold and a 1,000,000-node
hard cap by default; this permits finer meshes but does not require one million
nodes. Assign `h_u` to solid/island targets and the manifest's near-OBC target
to open chains. If budget-selected `h_u` exceeds the configured maximum size,
fail with an explicit request to raise that maximum; never silently lower the
bathymetry-supported floor. Preserve every source vertex and its lineage. For each source
chord of length \(L\), also derive the geometry-aware endpoint target
\(h_{\rm geo}=L\) and take the conservative minimum with the case target;
continue to report unavoidable short chords as `geometry_forced_subgrid`.
Keep sub-bathymetry-floor chord targets on the one-dimensional boundary trace;
the two-dimensional raster retains the bathymetry-supported interior floor.
Use wet-mask-aware raster sampling: retain bilinear interpolation only when
all four stencil cells are active; at a wet/dry interface select the
highest-weight positive-weight active corner instead of sharing an inactive
halo between wet banks. At a dry-cell centre use the coarsest covered fallback
so the wrong bank cannot inject a finer target. This interface guard is not a
global barrier-aware solver. When the boundary-trace wrapper reports no
positive-weight active raster support, make the trace authoritative and ignore
the covered-corner fallback for that query; record
`no_active_support_policy=boundary_trace_authoritative`. The fallback remains
the finite strict-coverage value for raster-only control runs and diagnostics.
Keep the node-budget integral on active wet cells only. Bound the raster
component with the active neighboring-stencil minimum and the trace component
with adaptive gradation-Lipschitz subcell lower bounds. Increase subcell
resolution only where the within-cell release is material relative to the
local target, up to \(32\times32\), so an isolated fine trace point is charged
to its influence area rather than a complete coarse raster cell.
Reconcile the immutable source boundary and rebuilt field for at most eight
fixed-point passes. The portfolio default uses `sampled_field`, so delivered
boundary targets follow the same pointwise callback. With default geometry
continuity enabled, the sampled chord-derived trace remains part of that
callback and preserves its subfloor targets. The standalone reconciler retains
an explicit `minimum` mode for source-target-lock workflows.

Use gradation `0.10` and boundary/field compatibility factor `1.5` for this
portfolio policy. Restore the `fvcom_boundary_trace_sampler_v2` sampled
approximation to the continuous trace over the cell-centred raster. Distribute
samples by linear endpoint-target metric length, include every delivered
endpoint and midpoint, and fail safely above five million samples. Each query
starts from 16 nearest samples and expands until a global lower bound proves
the exact minimum over that deterministic sample set. Process requested
locations in deterministic batches of at most 4,096 so QA memory does not grow
as `query_count × expanded_neighbor_count`; batching must not change values or
operational counters. The trace uses straight Euclidean distance: record
`barrier_aware=false` and the possible through-land/island refinement risk,
but do not claim observed leakage without a separate barrier audit. Recompute
node-budget preflight from the final callback sampled at every active wet
raster-cell centre, not from the stored raster alone. Then require the
realized boundary-edge/incident-first-ring symmetric scale ratio to have p95
at most `1.5` and maximum at most `2.0`, in addition to the edge-aware
boundary/first-ring/transition/interior `L/h` gates.
Record source and delivered boundary-node counts plus the adapter's boundary
discretization mode. Treat unmatched discretization policies as a
generator-plus-boundary-policy comparison, not an algorithm-only ranking, and
flag any conflict between the one-dimensional boundary target and the
two-dimensional `L/h` audit target.

Route by capability. The current clean-room adapter supports zero or one
noncyclic OBC, while the Gmsh adapter also supports plural and cyclic OBCs.
The operational raw default is Gmsh Frontal-Delaunay algorithm 6 alone.
Delaunay algorithm 5, the clean-room route, and MeshAdapt algorithm 1 run only
when explicitly named for a research comparison. Treat this as routing policy
rather than a composite quality winner; a failed Gmsh-6 run stops without
generator fallback.
The current continuity experiment is a no-conditioning `RAW` comparison:
disable OMA conditioning and postprocessing, while recording native generator
settings as raw provenance.
Record unsupported pairings as capability evidence rather than quality
failures. Compare metrics without a composite winner, and do not promote a
Gmsh candidate into the production path until the complete topology matrix
passes. Read `references/mesher_portfolio.md` for bundle, stage, fairness,
parallel-execution, and promotion rules.

Important controls:

- `--boundary-resolution-profile adaptive-coastal-v2`, default `adaptive-coastal-v2`. This is a deprecated compatibility selector; removed values are rejected and normal commands omit it.
- `--boundary-resolution-manifest`, required for new generation; its explicit per-OBC nodes and chains are authoritative.
- `--gradation`, default `0.20` in the production generator and `0.10` in `run_mesher_portfolio_case.py`; limits adjacent size growth after all candidates are combined.
- `--slope-elements 10` and `--coastal-distance-m 25000` control the nearshore bathymetric-slope candidate.
- `--hydraulic-elements-across-min 3` and `--hydraulic-elements-across-max 8` set the paired-bank cross-corridor element range. Importance increases the requested count continuously between those limits.
- `--hydraulic-max-width-m 20000` and `--hydraulic-bank-angle-deg 110` screen candidate bank pairs by span and opposition angle.
- `--hydraulic-longitudinal-gradation 0.10` limits size change along the accepted skeleton.
- `--obc-hold-distance-m 10000` holds the propagated OBC target over the first wet-domain reach. `--obc-transition-distance-m 60000` sets the minimum following quintic log-space transfer length; the workflow extends it to the gradation-required distance when necessary and records the requested, required, effective, and available distances.
- `--refine-iterations` controls adaptive insertion and `--smooth-iterations` controls the initial, fixed-boundary geometric smoothing that is part of mesh construction. These are distinct from the post-generation conditioning stages below.
- `--regional-spring-relaxation|--no-regional-spring-relaxation`; `minimal-topology-v1` disables this stage. Explicit mesh-conditioning research profiles retain their documented behavior.
- `--thin-triangle-repair|--no-thin-triangle-repair`; `minimal-topology-v1` uses only its bounded fixed-boundary topology repair. These switches retain their documented meaning for explicit mesh-conditioning research profiles.
- `--thin-repair-profile guarded-v1|systematic-v2|systematic-v3|systematic-v5|systematic-v6|none`, default `guarded-v1`; `systematic-v2` keeps boundary coordinates fixed, opt-in `systematic-v3` adds source-arc-only welding/sliding/redistribution, research-only `systematic-v5` couples persistent connectivity restriction and complete locked-star reconstruction to fixed-connectivity interaction bursts, and research-only `systematic-v6` adds coupled valence closure plus exact-zero relaxation entry.
- `--systematic-v3-obc-policy preserve|redistribute`, default `redistribute`; redistribution preserves OBC orientation and hard endpoints but may change its node set, which invalidates existing forcing and is recorded in `obc_remap_manifest.json`.
- `--systematic-v5-total-iterations`, `--systematic-v5-max-cycles`, `--systematic-v5-max-burst`, `--systematic-v5-thin-trigger`, `--systematic-v5-checkpoint-interval`, and `--systematic-v5-wall-time-s`; defaults are 1,000 iterations, six cycles, 250 iterations per burst, a 25-element thin trigger, 10-iteration checkpoints, and 21,600 seconds. V5 remains opt-in and never changes `auto`.
- `--systematic-v5-connectivity-restriction|--no-systematic-v5-connectivity-restriction` and `--systematic-v5-max-connectivity-transactions`, default enabled and 32; V5 and V6 use the same lineage-stable allowed-edge policy. Keep the feature research-only until a multi-region full-workflow matrix closes to zero debt under common defaults.
- `--systematic-v6-gate-policy strict-v6|topology-priority-v1|soft-topology-v1|topology-escrow-v1`, default `strict-v6`; selects one fixed whole-mesh closure policy. The adaptive ladder remains isolated in the research driver and never changes `auto`.
- `--systematic-v6-passage-removal|--no-systematic-v6-passage-removal`, default disabled; enables an explicitly authorized research-only topology delta. Generic V6 contains no case-specific passage node IDs.
- `--conditioning-profile auto|minimal-topology-v1|guarded-v1|aggressive-local-v2|none`; `auto` always resolves to `minimal-topology-v1`. This independent mesh-conditioning selector does not alter the mandatory Adaptive v2 boundary contract.
- `--minimal-conditioning-wall-time-s`, default 3,600 seconds; bounds the minimal profile per case. The profile also stops after four rounds, zero selected debt, or no accepted improvement.
- `--aggressive-conditioning-rounds`, `--aggressive-max-prunes-per-round`, and `--aggressive-max-valence-repairs-per-round`; bound repeated local edit/relax transactions.
- Standalone `repair_high_valence.py` additionally accepts `--max-valence-flip-batch` (default 64), `--max-valence-cluster-merges-per-round` (default 25), `--max-valence-l-over-h-count-increase` (default 0), and repeatable `--only-node-id-1based` targeting. Independent legal flips share one audit; adjacent zipper violations are attempted as one simple interior cavity before sequential single-node work. Keep the (L/h) trade budget at zero unless a documented hard-gate closure justifies it.
- `--aggressive-boundary-edit-policy kind-aware-envelope|split-only|none`; controls whether a non-hard boundary vertex may be removed inside a strict kind-specific shape/area envelope.
- `--area-transition-relaxation|--no-area-transition-relaxation`; normal generation applies sequential target-aware spring patches after thin repair.
- `--area-transition-max-patches`, default 12; bounds accepted local transition patches. The raw adjacent-area trigger defaults to `0.50`, equivalent to an area ratio of two.
- `--postprocess-profile`, compatibility default `none`; non-`none` integrated requests are rejected with standalone-tool guidance.
- `--max-total-nodes` and `--node-budget-stop-fraction` default to 1,000,000 and `0.90`, producing a 900,000-node planning gate. They audit the pre-triangulation estimate, including explicit boundary and boundary-front seeds. The audit is recorded for every boundary package and remains a hard gate for adaptive-coastal-v2. Independently audit the delivered node count after every generator; exceeding 1,000,000 is a hard failure even when preflight passed.
- `--land-spacing-m`, `--open-spacing-m`, `--max-interior-points`, and `--size-field-max-cells` set the boundary targets and execution limits. The default interior-seed ceiling is 900,000 so it does not retain the former 80,000-point bottleneck.
- `--bathy-fetch-halo-m`, default `2000`, buffers the assembled wet-domain
  polygon in projected meters before automatic CUDEM/NBS/CRM/ETOPO acquisition.
  RegionBPoly remains provenance and source-coverage intent, not the fetch
  extent. `open_boundary_bathymetry_support_incomplete` is a distinct pre-mesh
  failure.

To stop after the initial constrained mesh, explicitly use
`--no-regional-spring-relaxation --no-thin-triangle-repair
--thin-repair-profile none --conditioning-profile none
--no-area-transition-relaxation --postprocess-profile none`. Retain a positive
`--smooth-iterations` value when "initial mesh" includes the normal geometric
smoothing pass; set it to zero only when an unsmoothed triangulation is
specifically required. Verify every disabled stage's `reason` and operation
counts in `mesh_conditioning.json`; stage names and an empty topology ledger
alone are not proof that all movement stages were disabled.

The `fvcom_size_field_v4` production path first forms a segment-interpolated
solid-boundary background `h_S`. Inside the coastal mask it takes the pointwise
minimum of `h_S`, the bathymetric-gradient target `h_G`, and the paired-bank
hydraulic-corridor target `h_H`, then applies the lower gradation envelope to
obtain `h_N`. The hydraulic target uses width divided by a continuous
three-to-eight element count; the importance used in that count is a ranking
proxy based on wet-distance storage over cross-section area, not a solved
current velocity.

For an open domain, wet-domain graph propagation carries both distance from the
OBC and the originating delivered OBC target. The target is held for the
configured offshore reach and then blended to `h_N` with a quintic smootherstep
in log space. Apply the final eight-neighbor lower gradation envelope only over
wet cells. CFL remains diagnostic and never controls the size. Read
`references/fvcom_sms_quality.md` for the formulas, screening rules, diagnostics,
and interpretation limits.

## Generation-Time Conditioning

The default `minimal-topology-v1` order after constrained seeding/refinement is:

1. Hash/audit the mesh, canonical size field, bathymetry, boundary/OBC contract, and optional source metadata.
2. Repair valence above eight with every boundary coordinate and membership fixed. After each accepted valence transaction, immediately scan and repair any created or exposed superthin debt.
3. Repair residual connected superthin debt with protected-edge-safe flips, collapses, or bounded local cavity reconstruction. Never delete a triangle in isolation or remove a wet passage.
4. Repeat the terminal valence/thin scan for at most four rounds and accept only audited atomic retriangulations. Keep every regional-refinement metric as nonblocking debt. Roll back only structural failure or regression of the ordered valence/superthin debt; serialize every rejected candidate, boundary lineage, quality audit, edit ledger, and rollback manifest. Resample depth from the immutable bathymetry and repeat the complete serialization/quality audit.

The following is the explicit nondefault mesh-conditioning research order. It does not reactivate a removed boundary-resolution profile:

1. Recover all protected land, island, frame, and open-boundary edges.
2. Apply `spring-relax-v1` once to automatically selected poor-element patches. Keep physical boundary nodes fixed, keep connectivity fixed, and accept only backtracked force steps that preserve positive areas and do not regress controlled quality tails.
3. Apply `thin-repair-v1` to residual severe elements. Try legal nonprotected edge flips, then budgeted long interior-edge splits; relax only the edited patch and its graph halo.
4. In `aggressive-local-v2`, repeat local transactions in this order: remove target-redundant degree-3/4 interior vertices; process the selected thin-repair profile; repair every valence above eight with batched legal flips, connected zipper-cavity merges, or sequential cavity removal with distributed Steiner nodes. `guarded-v1` retains the quarantined flip/collapse/boundary-envelope ladder. Opt-in `systematic-v2` groups the extreme tail into lineage-stable components and tests two-to-four-ring local cavity reconstruction. `systematic-v3` retains v2, then projects boundary-directed interior apices onto their causal source arc, snaps them within `0.15h` or inserts them into the chain, tests deterministic 25/50/75/100-percent source-arc slides, and explicitly tests component boundary-node removal and protected-edge insertion. Research-only `systematic-v5` runs `connectivity restriction → locked-star V5 → component-cavity recovery → connectivity recheck → terminal locked-star closure`. The restriction stage ranks unprotected same-chain source-arc shortcuts, multi-superthin edges, and extreme `L/h`; reconstructs a complete constrained patch through the `1 → 2 → 4` ring ladder; and persists accepted forbidden edges by source lineage through compaction and later closure cycles. It never moves, inserts, or removes boundary nodes. Hard anchors never move, distinct passage banks never weld, and intermediate degenerate triangles are removed inside atomic transactions.
5. Apply `area-transition-relax-v1` to the worst excessive adjacent-area pairs one patch at a time. Always consider raw area change above 0.50; preempt inside steep target-size bands only when raw and target-normalized area mismatch are both excessive. Re-sample the Eulerian target field before every outer patch.
6. When `systematic-v2` or `systematic-v3` is selected, repeat it once after area-transition conditioning as terminal thin-debt closure. When V5 is selected, first establish a zero-superthin and zero-restricted-edge champion, then run bounded fixed-connectivity edge-angle-barrier bursts and the complete connectivity-restricted V5 closure after each selected checkpoint; compare quality only between zero-debt states and always finish with closure. V6 preserves the order `connectivity restriction → locked-star/multi-support repair → component cavity → connectivity recheck → terminal locked-star → optional authorized passage sweep → valence repair → immediate thin cleanup → second optional passage sweep → global audit`. Do not begin relaxation while superthin, valence-above-eight, or restricted-edge debt remains. On the standalone thin-repair interface, `--no-systematic-v5-boundary-window-fallback` makes every V5/V6 topology-preserving transaction boundary- and OBC-membership-immutable; it is not an integrated `run_fvcom_grid.py` option. V2 requires exact surviving boundary coordinates. V3/V5/V6 require zero source-arc normal offset, exact hard anchors, simple loops, passage-clearance loss no larger than 0.5 m, and a valid OBC remap. Roll back a complete relaxation/closure cycle unless it returns to zero debt, adds no singly connected or boundary-anomaly elements, and improves `q_l3_sigma` by at least `1e-4`.
   - When deterministic V5 reaches a plateau and the residual set is visually manageable, enter the research-only visual fallback instead of repeating the same blind ladder. Run `diagnose_superthin_components.py`, inspect every component image, and record an agent-reviewed `fvcom_visual_superthin_repair_plan_v1` for exactly one component. A JSON classification without actual image inspection is not visual review.
   - Apply the plan with `apply_visual_superthin_plan.py`. The plan is bound to the exact input SHA-256, names a bounded topology-tool sequence, and commits at most one globally audited transaction. Regenerate the component atlas after every accepted transaction before planning the next component because connectivity and component identifiers may change.
   - Visual mode has no fixed element-count trigger. Record why the complete residual inventory can be inspected and assigned bounded routes within the remaining time. Stop with explicit evidence when it is not manageable.
   - Visual routes may use constrained or protected-chord min-max cavity retriangulation, one or two inward-front or passage-centerline support nodes, optional reviewed support spokes, and last-resort source-arc insertion. Existing boundary coordinates and hard anchors remain exact. OBC insertion preserves the original ordered nodes but invalidates existing forcing and must emit an OBC remap manifest.
   - The current visual experiment targets zero superthin triangles only. Valence changes are recorded but are not a visual transaction gate, so a visual zero-superthin result is never by itself an FVCOM-readiness claim.
   - Keep whole-passage deletion outside visual repair and every normal automatic profile. Use it only after a reviewed narrow-passage case fails the bounded support/retriangulation routes and the user explicitly approves loss of wet connectivity. First run a diagnostic-only bilateral resolution-match test; do not insert a source-arc node in this branch. If one bank contains a target-relative over-resolution run at the causal apex, retain its normally spaced bracket nodes, remove the internal run together with the movable causal apex, and delete the complete incident triangle stars atomically.
   - A whole-passage plan must record the exact input hash, source-lineage removal set, retained bank brackets, intended wet-component and boundary-loop deltas, and the fact that the ordinary passage-preservation gate is being overridden. Commit only when those exact topology deltas occur, all areas remain positive, the delivered boundary graph is degree two, hard anchors and OBC order remain exact, restricted edges remain absent, and global superthin debt strictly decreases without quality/size/area-transition regression. Never continue deleting into a hard anchor merely to remove newly exposed singly connected elements; report that debt and stop.
7. Sample bathymetry at the delivered nodes, run FVCOM QA, and write the 2DM plus an edit ledger and terminal boundary lineage.

Hard anchors and all boundary nodes not explicitly handled by the kind-aware edit protocol remain fixed. Boundary splitting follows the existing arc exactly; boundary-vertex removal is forbidden at hard anchors and is accepted only inside the configured chord-deviation and cumulative domain-area envelopes. Treat unresolved boundary- or topology-imposed thin/transition defects as evidence for `needs_review`; never flatten the scientific target-size field or invoke a global Delaunay rebuild to force a pass. See `references/local_topology_conditioning.md` for the mathematics and transaction gates, and `references/six_case_minimal_conditioning_evidence_20260817.md` for the original matrix plus the corrected three-case gate-relaxation rerun.

## Standalone Tools

- `scripts/run_autonomous_thin_workflow.py`: run or resume the opt-in autonomous workflow. It performs minimal conditioning, produces the hash-bound diagnostic/decision stage, and resumes one selected component through local topology repair or boundary rebuild, Gmsh-6 remeshing, reconditioning, and the three independent closure/readiness decisions. A shoreline decision must bind its request-bounded CUSP extract; the command never pauses for human review.
- `scripts/manage_fvcom_grid_project.py`: initialize the fixed project tree, promote immutable stage selections, derive benchmark/submission decisions, automatically generate the standard review map, atomically publish terminal delivery artifacts, and validate readiness by exact hash.
- `scripts/diagnose_autonomous_thin.py`: create the whole-domain locator, local mesh/boundary and lineage diagrams, CUSP/GSHHS overlay, scale evidence, and one pending Codex decision per connected superthin component.
- `scripts/run_autonomous_thin_closure.py`: validate and execute one completed Codex decision. It preflights at most three boundary candidates, retains rejections, preserves protected and OBC lineage, and requires a complete Gmsh-6 remesh after any boundary transaction.
- `scripts/run_mesher_portfolio_case.py`: build one immutable regional `fvcom_size_field_v4` bundle and generate deterministic Gmsh Frontal-Delaunay-6 `RAW` by default under one node budget. Clean-room and Gmsh 1/5 require explicit `--candidates` values. It writes a metric-by-metric comparison and never claims common conditioning.
- `scripts/research/gmsh/prepare_lake_superior_boundary.py`: estimate first, then build a fresh zero-OBC GSHHG L2/L3 Lake Superior boundary package with the deterministic St. Marys numerical land gate.
- `scripts/research/gmsh/prepare_lake_superior_bathymetry.py`: convert a request-bounded ETOPO elevation mosaic to positive-down Lake Superior depth using the accepted wet-domain mask and the explicit 183.2 m IGLD 1985/EGM2008 caveat. Start estimate-first fetches from the committed `scripts/research/gmsh/continuity_cases/lake_superior_etopo_request.json`, and build a fresh ETOPO-only source index with the `cudem-bathy` connector instead of depending on a dated workspace index.
- `scripts/research/gmsh/validate_lake_superior_preparation.py`: independently hash and validate the boundary, island inventory, numerical gate, wet mask, depth conversion, and readiness selected by `scripts/research/gmsh/continuity_cases/lake_superior.json`. The current selector is boundary preparation v5 plus bathymetry v2; earlier preparation versions are superseded evidence, not fallback inputs. See the research Gmsh README for the immutable end-to-end command sequence.
- `scripts/plot_raw_mesh_transition.py`: read one raw portfolio 2DM and its locked boundary/field/quality artifacts, then write immutable whole-mesh and boundary/first-ring `L/h`, configured boundary/field-interface, and adjacent-area-change maps without editing the mesh.
- `scripts/run_portfolio_conditioning.py`: apply the same profile resolver to an arbitrary raw 2DM, preserve zero/single/plural OBC chains through lineage, accept external cyclicity/source-forcing metadata, and resample final depths from the locked bathymetry. A cyclic sidecar preserves evidence but cannot make standalone SMS 2DM self-describing.
- `scripts/run_conditioning_campaign.py`: process a fresh ordered case manifest sequentially, continue after case failures, retain immutable input hashes and per-case evidence, and write campaign JSON/CSV/Markdown with separate minimal-local-closure and full-readiness decisions.
- `scripts/diagnose_boundary_size_contract.py`: compare adaptive source-boundary targets with `fvcom_size_field_v4` at the same vertices and withhold triangle-algorithm attribution when the 1-D and 2-D targets conflict.
- `scripts/research/mesher_bakeoff/run_mesher_bakeoff.py`: lock a generator-neutral input bundle, plan non-overwriting candidate stages, execute `RAW` or `COMMON_CONDITIONED`, and compare only identical bundle/QA hashes. Unsupported topology/adapter pairs remain explicit and unranked.
- `scripts/analyze_mesh_quality.py`: analyze an existing `.2dm` without altering it.
- `scripts/relax_mesh_region.py`: apply the boundary-fixed spring solver to automatic defect patches or a requested lon-lat bounding box.
- `scripts/repair_thin_triangles.py`: apply `guarded-v1`, `systematic-v2`, `systematic-v3`, or closure-only `systematic-v5`/`systematic-v6`, or select `none` for a byte-preserving no-op copy. V3/V5/V6 use the shared connectivity-restriction policy and write an OBC remap/invalidation manifest.
- `scripts/restrict_superthin_connectivity.py`: audit ranked causal edges or run only the topology-preserving `superthin-connectivity-v1` repair. Supply mesh, boundary metadata, size field or uniform target, report, output mesh for repair, transaction budget, and wall-time limit.
- `scripts/fvcom_grid_generation/connectivity_restriction.py`: reuse the lineage-stable allowed-edge policy and read-only report schema `fvcom_superthin_connectivity_restriction_v1`.
- `scripts/fvcom_grid_generation/interaction_relaxation.py`: run the reusable fixed-connectivity edge-angle-area-barrier interaction engine; it never performs a global Delaunay rebuild.
- `scripts/fvcom_grid_generation/systematic_v5.py`: run the zero-debt champion, adaptive-burst, closure, rollback, plateau, recurrence, and deadline protocol used by V5 generation/local conditioning.
- `scripts/fvcom_grid_generation/systematic_v6.py`: run coupled topology-preserving closure, optional explicitly authorized passage work, valence repayment, exact-zero relaxation entry, and terminal source-contract audit.
- `scripts/fvcom_grid_generation/systematic_v6_policy.py`: reuse fixed V6 policy presets, adaptive research ordering, and bounded evidence-derived retry logic.
- `scripts/research/delaware/run_systematic_v6_overnight.py`: reproduce the frozen Delaware adaptive-policy experiment without placing case paths or node IDs in generic V6 defaults.
- `scripts/condition_mesh_local.py`: apply the complete `aggressive-local-v2` protocol to an existing 2DM, with optional adaptive boundary metadata and size field.
- `scripts/repair_high_valence.py`: run only the hard valence-repair branch; returns a nonzero status while any node valence exceeds eight.
- `scripts/diagnose_high_valence.py`: inventory every valence violation in one topology scan, classify its likely repair route, write CSV/GeoJSON/JSON, and plot domain-wide plus graph-local diagnostic maps; supply a conditioning report to plot rejected cases first.
- `scripts/diagnose_thinnest_triangles.py`: rank the lowest-quality delivered elements without editing the mesh, write per-edge flip/split blockers and gap-to-target evidence, and create a zoom panel plus individual figures for boundary, passage, junction, and interior-connectivity review.
- `scripts/diagnose_superthin_components.py`: create the mandatory connected-component visual atlas, including 1/2/4-ring patches, protected and open boundaries, hard anchors, local valence, passage evidence, and bounded candidate-point overlays; it also writes pending reviewed-plan templates.
- `scripts/diagnose_autonomous_thin.py`: extend the atlas with a complete-domain locator, boundary-lineage table, scale-derived CUSP request window, optional CUSP/GSHHS overlay, and pending hash-bound Codex decision documents.
- `scripts/run_autonomous_thin_closure.py`: validate one Codex decision, create an audited adaptive-boundary patch, rebuild an immutable Gmsh-6 case, optionally remesh and run `minimal-topology-v1`, and emit the three independent closure/readiness statuses. Never reuse a decision after any input hash changes.
- `scripts/apply_visual_superthin_plan.py`: validate and apply one SHA-bound, agent-reviewed component plan, write a checkpoint mesh/report/boundary/OBC remap, and reject stale, unreviewed, multi-component, structurally invalid, or non-improving transactions.
- `scripts/plot_superthin_component_node_ids.py`: render one component's complete 1/2/4-ring neighborhood with plain one-based source 2DM node identifiers for human passage-core selection.
- `scripts/apply_thin_passage_removal.py`: apply a SHA-bound sequence of human-approved or learned resolution-cluster whole-passage cuts, checkpoint every transaction, rebuild delivered boundary loops, and audit the explicitly intended topology deltas. This research-only tool is never called by `auto`.
- `scripts/replot_thin_passage_removal.py`: regenerate before/after passage-cut evidence from accepted checkpoint meshes and compact source-lineage manifests without replaying topology edits.
- `scripts/prune_redundant_vertices.py`: run only target-aware degree-3/4 interior-vertex pruning.
- `scripts/postprocess_fvcom_mesh.py`: run `rpw2019` or `projection-medium` cleanup explicitly for research.
- `scripts/compare_mesh_quality.py`: compare any two compatible quality JSON documents.

## Normal Outputs

- `fvcom_grid.2dm`
- `fvcom_grid_manifest.json`
- `mesh_quality.json`
- `mesh_conditioning.json`
- `mesh_edit_ledger.json` (operation, source-node lineage, and local edit evidence)
- `obc_remap_manifest.json` (original/delivered OBC lineage, source-arc position, redistribution status, and forcing compatibility)
- `boundary_nodes.geojson` (input boundary-node package)
- `delivered_boundary_nodes.geojson` (terminal constraint chains, including any recovery nodes)
- `size_field.nc`, `size_field.png`, and `size_field_components.png`
- `node_budget_preflight.json` and `node_budget_delivered.json` for every run, plus `boundary_contract_v2.json` for adaptive v2
- `mesh_nodes_elements.gpkg`
- `mesh_quality_elements.gpkg`
- `mesh_review_map.png`
- `mesh_review_map_manifest.json`
- progress JSON/JSONL artifacts

A mesher-portfolio run additionally writes the immutable input-bundle
manifest/hash, capability routing, one isolated candidate manifest per
generator/algorithm, and `raw_metric_comparison.json/.csv`. The generic
bakeoff layer writes immutable `RAW` and `COMMON_CONDITIONED` stage results
plus a no-composite comparison.

The v8 manifest records the size-field method and hydraulic-skeleton diagnostics, selected thin profile, aggressive local topology conditioning, area-transition conditioning, OBC remapping/forcing compatibility, hard valence status, and terminal lineage separately from `postprocess.enabled: false`.

## Acceptance

Emit separate policy decisions. `minimal_local_debt_closed` requires valence `<=8`, zero unique superthin triangles (`q<0.10` or minimum angle below `5°`), zero restricted-edge violations, and no structural regression. `benchmark_grid_baseline_ready` additionally requires finite positive depths, positive areas, one intended connected manifold mesh, traversable preserved constraints and OBC/exterior lineage, node-cap compliance, and exact 2DM roundtrip. `fvcom_ready` and `accepted` are compatibility aliases of that benchmark decision. Record ordinary angle tails, `q_l3_sigma`, area transition, bathymetric slope, `L/h`, boundary continuity, and nonstructural singly connected elements under `regional_refinement_debt`; they never veto the baseline. `submission_eligible` additionally requires forcing compatibility, self-describing OBC metadata, project provenance, and exact final hashes. Retain every terminal mesh and status with `needs_review` when the baseline cannot close.

For the visual superthin experiment, additionally require strict global superthin-count reduction, non-increasing superthin severity, no new residual component outside the reviewed lineage neighborhood, unchanged existing boundary coordinates, and refreshed visual evidence after every accepted component. Report `visual_zero_superthin_pass` separately from FVCOM readiness; use `visual_zero_superthin_pass_forcing_remap_required` when an OBC insertion occurred.

For a human-approved whole-passage deletion, replace the ordinary one-wet-component and passage-preservation gates only with the exact wet-component and boundary-loop deltas recorded in the plan. All other structural and serialization gates remain active. Report zero-superthin method success separately, and always mark the mesh non-FVCOM-ready while multiple wet components, new singly connected elements, valence above eight, `q_l3_sigma <= 0.75`, or forcing invalidation remains.

## Validation

```powershell
python scripts/selftest_fvcom_grid.py
python scripts/selftest_node_budget_policy.py
python scripts/selftest_boundary_contract_v2.py
python scripts/selftest_size_field.py
python scripts/selftest_gmsh_mesher_portfolio.py
python scripts/selftest_mesher_bakeoff.py
python scripts/selftest_portfolio_case.py
python scripts/selftest_portfolio_conditioning.py
python scripts/selftest_minimal_conditioning.py
python scripts/selftest_conditioning_campaign.py
python scripts/selftest_boundary_size_contract.py
python scripts/selftest_boundary_size_reconciliation.py
python scripts/selftest_edge_size_audit.py
python scripts/selftest_raw_transition_diagnostics.py
python scripts/selftest_local_topology_v2.py
python scripts/selftest_connectivity_restriction.py
python scripts/selftest_local_topology_v5_extensions.py
python scripts/selftest_systematic_v6.py
python scripts/selftest_visual_superthin.py
python scripts/selftest_grid_project.py
python scripts/selftest_grid_quality_policy.py
python scripts/selftest_mesh_review_map.py
python scripts/selftest_upstream_subworkflows.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
