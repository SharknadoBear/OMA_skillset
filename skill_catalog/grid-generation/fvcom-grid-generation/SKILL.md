---
name: fvcom-grid-generation
description: Generate FVCOM-ready SMS 2DM meshes from legacy or adaptive fvcom-bdry-arc packages and CUDEM/NBS/CRM/ETOPO bathymetry using clean-room constrained refinement, offshore-efficient sizing, regional spring relaxation, aggressive local topology conditioning, target-aware area-transition conditioning, and boundary-aware thin-triangle repair. Use when Codex needs explicit boundary-chain ingestion, adaptive nearshore-to-offshore size fields, variable-density seeding, regional mesh conditioning, hard FVCOM valence repair, ordered OBC nodestrings, FVCOM mesh QA, or small synthetic grid smoke tests; standalone broad OceanMesh-style cleanup remains available for research.
---

# fvcom-grid-generation

Use this skill after the boundary and bathymetry steps. Channel/thalweg
extraction is a reusable upstream analysis:

```text
fvcom-region-bpoly -> fvcom-bdry-arc -> cudem-bathy -> topobathy-flownet -> fvcom-grid-generation
```

## Core Rules

- Reuse upstream domain and boundary artifacts; do not redesign the region here.
- Prefer an adaptive boundary-resolution manifest when supplied. Otherwise preserve the legacy loop workflow.
- Keep full bathymetry for final node sampling and bound only the in-memory size-field grid.
- Use the same unified size algorithm for open and closed domains. Open domains blend the open- and solid-boundary targets; closed domains use the solid-boundary distance background. Apply only the bathymetric-slope and optional flow-network channel candidates inside the coastal/estuarine influence zone.
- Accept a passing `topobathy-flownet` manifest or generate it under the FVCOM run's `upstream/topobathy_flownet/` directory. Disabling flow-network extraction omits only the channel candidate.
- End normal adaptive generation after guarded shape relaxation, first-pass thin repair, aggressive local topology conditioning, target-aware area-transition relaxation, and a terminal constraint audit. Do not run the broad legacy postprocessor implicitly.
- Keep depths finite and positive down and treat OceanMesh2D GPL material as a method reference only.

## Primary Workflow

Adaptive package:

```powershell
python scripts/run_fvcom_grid.py --bdry-arc-manifest bdry_arc_manifest.json --boundary-loops-gpkg model_boundary_loops.gpkg --boundary-resolution-manifest boundary_resolution_manifest.json --boundary-resolution-profile adaptive-coastal-v2 --bathy-nc bathy.nc --run-dir runs/case --name case --mode test --postprocess-profile none
```

Legacy packages continue to use `--boundary-loops-gpkg` without a resolution manifest.

Important controls:

- `--boundary-resolution-profile legacy|adaptive-coastal-v1|adaptive-coastal-v2`, default `legacy`. V2 is opt-in and requires an upstream `pass` manifest before gridding.
- `--boundary-resolution-manifest`, optional; explicit nodes and chains take precedence over legacy densification.
- `--gradation`, default `0.20`; limits adjacent size growth after all candidates are combined.
- `--slope-elements 10` and `--coastal-distance-m 12000` control the nearshore bathymetric-slope candidate.
- `--channel-flownet-manifest` consumes an existing passing `topobathy-flownet` package. Otherwise `--channel-flownet` (enabled by default) runs that skill on the original bathymetry NetCDF and the exact model-domain polygon, including holes. Use `--no-channel-flownet` only when the channel candidate is intentionally unavailable.
- `--channel-flownet-source-area-km2`, default `1.0`, and optional `--channel-flownet-target-resolution-m` control extraction. `--channel-reslope-angle-deg 60`, `--channel-elements-per-depth 1`, and optional `--channel-min-size-m` control how accepted DHSVM `SegOrder` arcs become a channel-size candidate.
- `--regional-spring-relaxation|--no-regional-spring-relaxation`; normal generation applies one guarded, defect-selected spring-equilibrium stage by default.
- `--thin-triangle-repair|--no-thin-triangle-repair`; normal generation applies local protected-edge-safe flips/splits and patch relaxation by default.
- `--thin-repair-profile guarded-v1|systematic-v2|systematic-v3|systematic-v5|systematic-v6|none`, default `guarded-v1`; `systematic-v2` keeps boundary coordinates fixed, opt-in `systematic-v3` adds source-arc-only welding/sliding/redistribution, research-only `systematic-v5` couples persistent connectivity restriction and complete locked-star reconstruction to fixed-connectivity interaction bursts, and research-only `systematic-v6` adds coupled valence closure plus exact-zero relaxation entry.
- `--systematic-v3-obc-policy preserve|redistribute`, default `redistribute`; redistribution preserves OBC orientation and hard endpoints but may change its node set, which invalidates existing forcing and is recorded in `obc_remap_manifest.json`.
- `--systematic-v5-total-iterations`, `--systematic-v5-max-cycles`, `--systematic-v5-max-burst`, `--systematic-v5-thin-trigger`, `--systematic-v5-checkpoint-interval`, and `--systematic-v5-wall-time-s`; defaults are 1,000 iterations, six cycles, 250 iterations per burst, a 25-element thin trigger, 10-iteration checkpoints, and 21,600 seconds. V5 remains opt-in and never changes `auto`.
- `--systematic-v5-connectivity-restriction|--no-systematic-v5-connectivity-restriction` and `--systematic-v5-max-connectivity-transactions`, default enabled and 32; V5 and V6 use the same lineage-stable allowed-edge policy. Keep the feature research-only until a multi-region full-workflow matrix closes to zero debt under common defaults.
- `--systematic-v6-gate-policy strict-v6|topology-priority-v1|soft-topology-v1|topology-escrow-v1`, default `strict-v6`; selects one fixed whole-mesh closure policy. The adaptive ladder remains isolated in the research driver and never changes `auto`.
- `--systematic-v6-passage-removal|--no-systematic-v6-passage-removal`, default disabled; enables an explicitly authorized research-only topology delta. Generic V6 contains no case-specific passage node IDs.
- `--conditioning-profile auto|guarded-v1|aggressive-local-v2|none`; `auto` promotes adaptive packages to `aggressive-local-v2` and retains guarded legacy behavior.
- `--aggressive-conditioning-rounds`, `--aggressive-max-prunes-per-round`, and `--aggressive-max-valence-repairs-per-round`; bound repeated local edit/relax transactions.
- Standalone `repair_high_valence.py` additionally accepts `--max-valence-flip-batch` (default 64), `--max-valence-cluster-merges-per-round` (default 25), `--max-valence-l-over-h-count-increase` (default 0), and repeatable `--only-node-id-1based` targeting. Independent legal flips share one audit; adjacent zipper violations are attempted as one simple interior cavity before sequential single-node work. Keep the (L/h) trade budget at zero unless a documented hard-gate closure justifies it.
- `--aggressive-boundary-edit-policy kind-aware-envelope|split-only|none`; controls whether a non-hard boundary vertex may be removed inside a strict kind-specific shape/area envelope.
- `--area-transition-relaxation|--no-area-transition-relaxation`; normal generation applies sequential target-aware spring patches after thin repair.
- `--area-transition-max-patches`, default 12; bounds accepted local transition patches. The raw adjacent-area trigger defaults to `0.50`, equivalent to an area ratio of two.
- `--postprocess-profile`, compatibility default `none`; non-`none` integrated requests are rejected with standalone-tool guidance.
- `--max-total-nodes` and `--node-budget-stop-fraction` audit the pre-triangulation estimate, including explicit boundary and boundary-front seeds. The audit is recorded for every boundary package and remains a hard gate for adaptive-coastal-v2.
- `--land-spacing-m`, `--open-spacing-m`, `--max-interior-points`, and `--size-field-max-cells` set the boundary targets and execution limits.

The unified background is continuous. For an open domain, compute exact segment
distances and targets for the open and solid boundary families, form
`phi = d_open / (d_open + d_solid)`, apply cubic smoothstep, and blend the two
targets in log space. For a closed domain, grade outward from the solid
boundary target. Inside the coastal mask, take the minimum of this background,
the bathymetric-slope candidate, and the optional DHSVM flow-network channel
candidate. Outside that mask, retain only the boundary background. Finally clip
to physical bounds and apply the eight-neighbor lower gradation envelope. CFL
remains diagnostic and never controls the size.

## Generation-Time Conditioning

Use this order after constrained seeding/refinement:

1. Recover all protected land, island, frame, and open-boundary edges.
2. Apply `spring-relax-v1` once to automatically selected poor-element patches. Keep physical boundary nodes fixed, keep connectivity fixed, and accept only backtracked force steps that preserve positive areas and do not regress controlled quality tails.
3. Apply `thin-repair-v1` to residual severe elements. Try legal nonprotected edge flips, then budgeted long interior-edge splits; relax only the edited patch and its graph halo.
4. In `aggressive-local-v2`, repeat local transactions in this order: remove target-redundant degree-3/4 interior vertices; process the selected thin-repair profile; repair every valence above eight with batched legal flips, connected zipper-cavity merges, or sequential cavity removal with distributed Steiner nodes. `guarded-v1` retains the quarantined flip/collapse/boundary-envelope ladder. Opt-in `systematic-v2` groups the extreme tail into lineage-stable components and tests two-to-four-ring local cavity reconstruction. `systematic-v3` retains v2, then projects boundary-directed interior apices onto their causal source arc, snaps them within `0.15h` or inserts them into the chain, tests deterministic 25/50/75/100-percent source-arc slides, and explicitly tests component boundary-node removal and protected-edge insertion. Research-only `systematic-v5` runs `connectivity restriction → locked-star V5 → component-cavity recovery → connectivity recheck → terminal locked-star closure`. The restriction stage ranks unprotected same-chain source-arc shortcuts, multi-superthin edges, and extreme `L/h`; reconstructs a complete constrained patch through the `1 → 2 → 4` ring ladder; and persists accepted forbidden edges by source lineage through compaction and later closure cycles. It never moves, inserts, or removes boundary nodes. Hard anchors never move, distinct passage banks never weld, and intermediate degenerate triangles are removed inside atomic transactions.
5. Apply `area-transition-relax-v1` to the worst excessive adjacent-area pairs one patch at a time. Always consider raw area change above 0.50; preempt inside steep target-size bands only when raw and target-normalized area mismatch are both excessive. Re-sample the Eulerian target field before every outer patch.
6. When `systematic-v2` or `systematic-v3` is selected, repeat it once after area-transition conditioning as terminal thin-debt closure. When V5 is selected, first establish a zero-superthin and zero-restricted-edge champion, then run bounded fixed-connectivity edge-angle-barrier bursts and the complete connectivity-restricted V5 closure after each selected checkpoint; compare quality only between zero-debt states and always finish with closure. V6 preserves the order `connectivity restriction → locked-star/multi-support repair → component cavity → connectivity recheck → terminal locked-star → optional authorized passage sweep → valence repair → immediate thin cleanup → second optional passage sweep → global audit`. Do not begin relaxation while superthin, valence-above-eight, or restricted-edge debt remains. `--no-systematic-v5-boundary-window-fallback` makes every V5/V6 topology-preserving transaction boundary- and OBC-membership-immutable. V2 requires exact surviving boundary coordinates. V3/V5/V6 require zero source-arc normal offset, exact hard anchors, simple loops, passage-clearance loss no larger than 0.5 m, and a valid OBC remap. Roll back a complete relaxation/closure cycle unless it returns to zero debt, adds no singly connected or boundary-anomaly elements, and improves `q_l3_sigma` by at least `1e-4`.
   - When deterministic V5 reaches a plateau and the residual set is visually manageable, enter the research-only visual fallback instead of repeating the same blind ladder. Run `diagnose_superthin_components.py`, inspect every component image, and record an agent-reviewed `fvcom_visual_superthin_repair_plan_v1` for exactly one component. A JSON classification without actual image inspection is not visual review.
   - Apply the plan with `apply_visual_superthin_plan.py`. The plan is bound to the exact input SHA-256, names a bounded topology-tool sequence, and commits at most one globally audited transaction. Regenerate the component atlas after every accepted transaction before planning the next component because connectivity and component identifiers may change.
   - Visual mode has no fixed element-count trigger. Record why the complete residual inventory can be inspected and assigned bounded routes within the remaining time. Stop with explicit evidence when it is not manageable.
   - Visual routes may use constrained or protected-chord min-max cavity retriangulation, one or two inward-front or passage-centerline support nodes, optional reviewed support spokes, and last-resort source-arc insertion. Existing boundary coordinates and hard anchors remain exact. OBC insertion preserves the original ordered nodes but invalidates existing forcing and must emit an OBC remap manifest.
   - The current visual experiment targets zero superthin triangles only. Valence changes are recorded but are not a visual transaction gate, so a visual zero-superthin result is never by itself an FVCOM-readiness claim.
   - Keep whole-passage deletion outside visual repair and every normal automatic profile. Use it only after a reviewed narrow-passage case fails the bounded support/retriangulation routes and the user explicitly approves loss of wet connectivity. First run a diagnostic-only bilateral resolution-match test; do not insert a source-arc node in this branch. If one bank contains a target-relative over-resolution run at the causal apex, retain its normally spaced bracket nodes, remove the internal run together with the movable causal apex, and delete the complete incident triangle stars atomically.
   - A whole-passage plan must record the exact input hash, source-lineage removal set, retained bank brackets, intended wet-component and boundary-loop deltas, and the fact that the ordinary passage-preservation gate is being overridden. Commit only when those exact topology deltas occur, all areas remain positive, the delivered boundary graph is degree two, hard anchors and OBC order remain exact, restricted edges remain absent, and global superthin debt strictly decreases without quality/size/area-transition regression. Never continue deleting into a hard anchor merely to remove newly exposed singly connected elements; report that debt and stop.
7. Sample bathymetry at the delivered nodes, run FVCOM QA, and write the 2DM plus an edit ledger and terminal boundary lineage.

Hard anchors and all boundary nodes not explicitly handled by the kind-aware edit protocol remain fixed. Boundary splitting follows the existing arc exactly; boundary-vertex removal is forbidden at hard anchors and is accepted only inside the configured chord-deviation and cumulative domain-area envelopes. Treat unresolved boundary- or topology-imposed thin/transition defects as evidence for `needs_review`; never flatten the scientific target-size field or invoke a global Delaunay rebuild to force a pass. See `references/local_topology_conditioning.md` for the mathematics and transaction gates.

## Standalone Tools

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
- `size_field.nc` and `size_field.png`
- `node_budget_preflight.json` for every run and `boundary_contract_v2.json` for adaptive v2
- `upstream/topobathy_flownet/products/run_manifest.json` and its DHSVM flow-line package when run-local extraction is enabled
- `mesh_nodes_elements.gpkg`
- `mesh_quality_elements.gpkg`
- `mesh_review_map.png`
- progress JSON/JSONL artifacts

The v7 manifest records the selected thin profile, aggressive local topology conditioning, area-transition conditioning, OBC remapping/forcing compatibility, hard valence status, and terminal lineage separately from `postprocess.enabled: false`.

## Acceptance

Require a successful 2DM roundtrip with matching connectivity and valid OBC order, finite positive depths, positive-area elements, complete constraints, one manifold component, exact hard anchors, sub-centimeter text-serialization shifts, and valid exterior/island loops. V1/v2 require exact surviving boundary coordinates; v3/v5/v6 require delivered boundary vertices on their original source polylines and a complete OBC remap. Conditioning must not materially regress controlled lower-tail metrics, stage-baseline `L/h` (explicit 0.1% numerical tolerance), or area-transition defect counts. Require `q_l3_sigma > 0.75`, zero superthin triangles (`q<0.10` or minimum angle below `5°`), zero restricted-edge violations, and true vertex-neighbor valence `<=8` for FVCOM readiness. Three consecutive improving zero-debt checkpoints demonstrate steady quality growth; a partial thin reduction does not. Retain artifacts with `needs_review` when a legal guarded edit cannot close a gate. Keep case-specific controlled-proof checks in their research notes rather than universal acceptance.

For the visual superthin experiment, additionally require strict global superthin-count reduction, non-increasing superthin severity, no new residual component outside the reviewed lineage neighborhood, unchanged existing boundary coordinates, and refreshed visual evidence after every accepted component. Report `visual_zero_superthin_pass` separately from FVCOM readiness; use `visual_zero_superthin_pass_forcing_remap_required` when an OBC insertion occurred.

For a human-approved whole-passage deletion, replace the ordinary one-wet-component and passage-preservation gates only with the exact wet-component and boundary-loop deltas recorded in the plan. All other structural and serialization gates remain active. Report zero-superthin method success separately, and always mark the mesh non-FVCOM-ready while multiple wet components, new singly connected elements, valence above eight, `q_l3_sigma <= 0.75`, or forcing invalidation remains.

## Validation

```powershell
python scripts/selftest_fvcom_grid.py
python scripts/selftest_boundary_contract_v2.py
python scripts/selftest_size_field.py
python scripts/selftest_local_topology_v2.py
python scripts/selftest_connectivity_restriction.py
python scripts/selftest_local_topology_v5_extensions.py
python scripts/selftest_systematic_v6.py
python scripts/selftest_visual_superthin.py
python -m compileall scripts
python C:\Users\huan111\.codex\skills\.system\skill-creator\scripts\quick_validate.py .
```
