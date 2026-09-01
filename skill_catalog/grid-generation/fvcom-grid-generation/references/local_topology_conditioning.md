# Local Topology Conditioning

Use this note for `minimal-topology-v1`, `aggressive-local-v2`, `run_portfolio_conditioning.py`, `condition_mesh_local.py`, `repair_high_valence.py`, and `prune_redundant_vertices.py`.

The authoritative decision buckets and thresholds are defined only in
[`fvcom_grid_quality_policy_v1.json`](fvcom_grid_quality_policy_v1.json). This
note explains the algorithms; it does not define a competing readiness policy.

## Minimal topology v1

`auto` resolves to `minimal-topology-v1` in both integrated generation and standalone portfolio conditioning. This profile fixes every original topological boundary coordinate and OBC membership, disables pruning, spring and micro-relaxation, area-transition relaxation, general boundary edits, passage deletion, and global retriangulation, and permits only atomic local retriangulation. Its sole boundary-discretization exception is a new midpoint on a causal non-OBC source-arc edge for the exact fixed-hard-fan class described below; no original boundary node moves or disappears.

Each of at most four rounds applies valence repair first, immediately scans and repairs any superthin debt created by an accepted valence transaction, repairs residual connected superthin debt with protected-edge-safe flips, collapses, or bounded local cavities, and finishes with another valence scan plus immediate thin cleanup. If one residual component is exactly one all-fixed/all-hard triangle on one non-OBC chain, test one midpoint insertion on its protected source-arc edge and constrained reconstruction through the `1 -> 2 -> 4` ring ladder. Use zero interior support nodes, skip every OBC-touching component, and commit the first smallest passing patch only when the global superthin tuple strictly decreases, maximum valence stays at most eight, all original boundary coordinates and hard anchors are exact, OBC lineage/order is identical, and the full structural audit passes. This exact automatic route does not require a visual review; ineligible or rejected residuals retain the existing diagnostic route. Stop on zero selected debt, no accepted improvement, or the per-case wall-clock deadline. Never delete one triangle as an isolated operation and never remove a wet passage.

Hash the raw mesh, canonical size field, immutable bathymetry, boundary/OBC contract, and source boundary metadata. For the default minimal and autonomous paths, accept transactions lexicographically: preserve absolute structural invariants first, reduce the valence tuple second, and reduce the superthin tuple third. A valence improvement may be escrowed while immediate superthin cleanup runs. A superthin repair may not worsen the retained valence tuple. Ordinary angle tails, `q_min`, `q_p01`, `q_{L3\sigma}`, adjacent-area transition, bathymetric slope, singly-connected count without a structural break, and `L/h` are Class-2 debt and never roll back a useful Class-1 repair. Explicit legacy and research profiles retain their documented experimental guards.

When a structural or ordered debt gate rejects the primary candidate, retain `candidate.2dm`, its quality JSON, boundary-node lineage, edit ledger, and a hash-bound rollback manifest under `rejected_primary_candidate/`. Deliver the best structurally valid lexicographic champion, not the mesh with the smoothest Class-2 statistics. Resample final positive-down depths and repeat the full 2DM roundtrip and quality audit.

Report `minimal_local_debt_closed` separately from `benchmark_grid_baseline_ready`. Both require valence at most eight and zero unique superthin triangles; the benchmark decision additionally requires the complete structural, boundary/OBC, bathymetry, node-cap, and serialization baseline. `fvcom_ready` and `accepted` are compatibility aliases of the benchmark decision. `submission_eligible` additionally requires forcing/OBC remap compatibility, project provenance, and exact final hashes. External plural/cyclic OBC metadata may preserve benchmark validity while leaving submission ineligible when the SMS 2DM alone is not self-describing.

## Geometry and hard limits

For triangle (T), use the FVCOM-favored equilateral quality

\[
q_T=\frac{4\sqrt{3}A_T}{L_1^2+L_2^2+L_3^2}.
\]

Flag a superthin triangle when (q_T<0.10) or its minimum angle is below (5^\circ). The hard FVCOM topology gate is true vertex-neighbor valence

\[
\nu_i=|\{j:(i,j)\in E\}|\le 8
\]

at every node. This is unique-neighbor valence, not incident-triangle count.

Let (h_i) be the local target spacing and use the harmonic edge target

\[
h_{ij}=\frac{2}{h_i^{-1}+h_j^{-1}}.
\]

Topology edits remain local, use the current Eulerian target field, and never invoke a global Delaunay rebuild.

## Edit ladder

Select `guarded-v1` for the established per-element ladder or opt-in `systematic-v2` for connected-component cavity repair. `systematic-v2` first retains legal movable-interior flips and collapses, then inventories remaining superthin triangles by lineage-stable connected component. It classifies each component as interior connectivity/transition, fixed-boundary fan, fixed-boundary hard-anchor fan, under-resolved passage, or mixed; expands the patch through two to four triangle-adjacency rings; and tests deterministic local reconstructions.

Systematic cavity candidates retain every patch-boundary and protected edge, may remove only movable interior component nodes, and may add interior support points. Fixed-boundary support points lie on the wet side of the protected segment. Passage candidates place paired or centerline support points and assign inserted nodes a recorded feature target no larger than half the measured passage width. Existing boundary coordinates, hard anchors, boundary chains, wet connected components, and OBC order are immutable. A candidate is committed only when it reduces the global superthin count and also preserves the ordinary quality, valence, size, area-transition, topology, and serialization gates. Blocked components and every rejected candidate remain explicit evidence; the algorithm never closes the passage to force a pass.

Select opt-in `systematic-v3` when the v2 debt is boundary-constrained. V3 retains the complete v2 ladder, then permits only source-arc-tangential boundary changes. A movable interior apex may be projected onto its causal protected segment, snapped to an existing boundary node within `0.15h`, or inserted into the chain. Zero-area records created by the weld are deleted before the transaction audit. Non-hard boundary windows extend no more than four chain positions from the component and stop at hard anchors or boundary-kind junctions. Target-equalized positions are tested at 25%, 50%, 75%, and 100% of the source-arc displacement, together with explicit non-hard vertex removal and protected-edge insertion candidates.

Hard anchors remain coordinate-exact. Passage banks retain distinct chain identity and may lose no more than 0.5 m of minimum clearance. OBC policy `redistribute` permits non-hard insertion/removal while preserving endpoints and orientation; every delivered OBC node receives retained/slid/redistributed/inserted lineage and source-arc position in `obc_remap_manifest.json`. Any coordinate or node-set change invalidates existing OBC forcing.

Select research-only `systematic-v5` when testing relaxation against complete local connectivity reconstruction. For the causal shortest-altitude apex, collect every incident triangle and its ordered perimeter, remove the complete old fan inside a trial, and test contraction, center elimination, center relocation, source-arc insertion, and one support-node fallback. Treat an open fixed-boundary star as a polygonal boundary fan: retain the hard center and every boundary coordinate while replacing all incident faces. Use deterministic ear triangulation and apply Lawson flips only to non-protected internal patch edges. Expand a failed patch through the explicit `1, 2, 4` ring ladder, then use connected-component cavity/support recovery. Never use a global Delaunay rebuild or erode the physical wet boundary.

Run `superthin-connectivity-v1` before the locked-star ladder. Inventory every unprotected edge touching a superthin component, ranking a same-chain shortcut first when both source-arc/chord and source-arc/local-target ratios are at least 3, followed by edges shared by multiple superthin triangles and extreme `L/h`. Test no more than eight causal edges per component and 32 accepted transactions per round. For each edge, combine its attached component and endpoint stars, then expand through `1 → 2 → 4` triangle rings. Lock the complete outer perimeter and split the patch into wet subfaces along internal protected land, island, and OBC segments. Retriangulate each subface with covered legal diagonals, reuse unchanged interior nodes, and apply protected-safe Lawson flips. Never move, insert, or remove a boundary node in this stage.

Persist every accepted forbidden edge as a sorted pair of source-node lineage IDs. Apply the same policy to ear clipping, unchanged-node insertion, cavity reconstruction, and Lawson legalization; remap it after compaction and every closure cycle. A transaction commits only when it removes the causal edge, reduces global superthin debt, preserves `q_l3_sigma`, first-percentile quality, valence debt, `L/h>1.55` count, area-transition count, all coordinates, chains, OBC membership/order, protected edges, and wet topology, and finishes with zero restricted-edge violations. Record chain positions, source-arc span, chord, patch mode, constrained subfaces, candidate failures, replacement topology, and audits in the edit ledger using schema `fvcom_superthin_connectivity_restriction_v1`.

Run V5 closure in this exact order: connectivity restriction, locked-star V5, connected-component cavity recovery, connectivity recheck, and terminal locked-star closure. Run the same complete closure before relaxation and after every interaction checkpoint. Keep the boundary-adaptive V3 fallback available, but disable it for a topology-only controlled proof. When `systematic_v5_enable_boundary_window_fallback` is false, do not generate locked-star source-arc insertion/snap modes and reject any transaction whose delivered OBC lineage sequence differs from the frozen source sequence. Use `restrict_superthin_connectivity.py --mode audit` for read-only diagnosis and `--mode repair` for the isolated topology-only stage.

V5 separates hard transaction gates from cycle gates. Check finite coordinates, positive delivered areas, manifold topology, the locked perimeter, protected edges, wet/island component counts, exact hard anchors, source-arc membership, passage-bank identity and clearance, boundary loops, OBC order, and the `1e-4` domain-area budget immediately. Defer `q_l3_sigma`, percentile tails, valence, `L/h`, area transitions, singly connected triangles, and boundary anomalies until relaxation, closure, and local recovery finish. A zero-debt checkpoint is eligible to become champion only when it adds no singly connected or boundary-anomaly element and improves `q_l3_sigma` by at least `1e-4`.

The reusable V5 interaction engine moves fixed-connectivity vertices under edge, equilateral-angle, and positive-area barrier forces. Keep all boundary and hard-anchor nodes exact during the burst. Stop on the configured unified superthin trigger, recurring repaired lineage, nonpositive geometry, three rejected steps, or two checkpoint gains below `1e-5`. Select the highest-`q_l3_sigma` checkpoint within the trigger, run exhaustive closure, and roll back the complete cycle on any failed gate. Always deliver the last accepted zero-debt champion with no subsequent relaxation. Default limits are six committed cycles and 1,000 cumulative iterations. V5 and V6 remain opt-in until a multi-region full-workflow matrix demonstrates repeatable exact-zero closure under identical defaults.

Treat wall-clock checks as safety gates, not advisory logging. Check before every relaxation iteration, component, candidate, audit, and serialization. Reserve enough time for one full-mesh audit; if an existing helper contains a non-preemptible audit that exceeds the reserve, record a hard-stop defect and do not begin policy screening from a nonzero or unreported closure.

## Agent-reviewed visual fallback

When a complete deterministic V5 pass accepts no transaction, a residual lineage recurs twice, or another blind pass would consume the terminal audit reserve, inventory the connected superthin components and decide whether every remaining mechanism can be visually inspected and assigned a bounded route. There is intentionally no numeric component threshold. Record the manageability judgment and its time/evidence basis.

Visual review means opening every component image and interpreting the mesh geometry. The atlas must show the selected triangles, one/two/four-ring patch limits, boundary kind, hard anchors, OBC and protected segments, restricted edges, local valence, target spacing, passage width, and candidate support or source-arc insertion points. Deterministic context labels are evidence for the review but do not replace it.

Write one `fvcom_visual_superthin_repair_plan_v1` per transaction. Bind it to the exact input mesh SHA-256, identify one component, preserve the visual observations, list a primary and bounded fallback topology-tool sequence, and require strict global superthin reduction. Reject stale or unreviewed plans. Apply only one component, audit the whole mesh, serialize the accepted checkpoint, and regenerate the atlas before reviewing the next residual.

Use the visual route implied by the observed mechanism: connectivity restriction and constrained patch reconstruction for an artificial shortcut; inward-front support and optional source-arc insertion for a fixed fan; free or reviewed-spoke paired-bank/centerline support for an under-resolved passage; and legal constrained or protected-chord min-max cavity retriangulation/support for an interior transition. Existing boundary coordinates and hard anchors do not move. A new OBC node lies on the source arc, preserves the original ordered OBC nodes as a subsequence, and marks existing forcing incompatible.

The autonomous visual path must retain the best valence tuple while removing superthin debt. Positive area, manifold wet topology, protected constraints, passage identity, source-arc membership, restricted-edge absence, and exact serialization remain mandatory. Quality tails, size/area transition, slope, and nonstructural singly-connected changes are recorded as regional debt and do not reject a structurally valid valence/superthin improvement.

## Human-approved whole-passage deletion

This is a research-only fallback for an artificial narrow wet connection that the user intends to remove. It is not another visual repair primitive and is never entered by an automatic profile. Before deletion, use bilateral source-arc resolution only as a diagnostic: compare edge length to the harmonic target on both banks and identify a bounded run with `L/h < 0.55` at the causal apex. Do not add a boundary node in this branch. Prior support or source-arc trials that retain or increase local superthin debt are admissible evidence for entering the fallback.

Case-specific learned passage sets belong in a research driver, never in the reusable V6 defaults. Delete the union of all triangles incident to an explicitly approved node set, remove those nodes, compact by source lineage, and rebuild every delivered boundary loop from topological boundary edges. The selected superthin component must be contained in the deleted star union. The frozen Delaware reproduction details are isolated in `references/research_delaware_systematic_v6.md`.

For a new component, retain normally spaced bracket nodes around the closest contiguous over-resolved bank run, infer only its internal nodes, and add a movable component apex when present. Test the learned set transactionally; do not expand blindly. The plan must explicitly state the expected change in wet connected components and boundary-loop components. A same-loop cut may split the wet mesh; a cut joining two island banks may merge their boundary loops without changing the wet-component count. These topology changes are accepted only when they exactly match the reviewed plan.

Require positive areas, no nonmanifold edges, a degree-two delivered boundary graph, exact surviving hard anchors, ordered and unchanged OBC membership, zero unused/repeated/duplicate triangles, zero restricted-lineage edges, strict global superthin reduction, non-increasing severity, and non-regression of `q_min`, `q_p01`, minimum angle, `L/h > 1.55`, and area-transition counts. Serialize at 12 decimals and replay deterministically from the frozen hash. Newly exposed singly connected triangles are reported. Stop instead of peeling when further deletion would propagate into hard anchors or outside the reviewed passage core.

Whole-passage zero-superthin closure is method success only. A mesh with multiple wet components, singly connected elements, valence above eight, `q_l3_sigma <= 0.75`, or forcing invalidation is not FVCOM-ready.

1. **Redundant degree-3/4 vertex pruning.** An unfixed interior node with an ordered 3- or 4-node ring is removable only when a legal cavity triangulation has positive signed area, maximum (L/h\le1.25), minimum (q_T\ge0.40), minimum angle at least (28^\circ), and no resulting neighbor exceeds valence eight. For a quadrilateral, evaluate both diagonals and lexicographically favor smaller maximum (L/h), larger minimum quality, then larger minimum angle.

2. **Interior superthin-pair collapse.** If two superthin triangles share an unprotected interior edge (e=(a,b)), both endpoints are movable, the simplicial link condition holds, and (L_{ab}/h_{ab}\le0.50), merge (a) and (b) at

   \[
   x_{ab}=\tfrac12(x_a+x_b).
   \]

   Delete the two degenerate incident records created by the merge, compact the cavity, and retain source-node lineage in the edit ledger.

3. **Boundary superthin repair.** Work through a severity-ordered queue and quarantine a rejected candidate so it cannot terminate work on independent defects. First recognize a redundant boundary ear whose deletion exposes the existing chord without losing a protected source arc; charge its actual global signed-area difference to the domain-area budget and require the resulting boundary graph to remain traversable. For a coarse boundary arc with an interior apex close to that arc, project the apex onto the immutable source segment and insert it in the ordered chain. The weld must satisfy target-relative distance and altitude limits, kind-specific absolute displacement, hard-anchor and landfall buffers, and channel-clearance guards; then re-sample its target from the Eulerian size field. Other protected boundary sides may still be split. A non-hard boundary vertex may be removed only inside the boundary envelope. Removal is accepted only when the vertex-to-neighbor chord deviation is below

   \[
   \delta_k=\min(\delta_{k,\mathrm{abs}},f_k\bar h),
   \]

   using land/island defaults ((5\ \mathrm{m},0.03)) and open-boundary defaults ((250\ \mathrm{m},0.05)), neighboring boundary kinds agree, the node is not a hard anchor, and cumulative domain-area change remains below (10^{-4}) of initial area. `split-only` and `none` disable successively more permissive branches. The minimum is intentional: the absolute cap and target-relative cap must both be satisfied.

4. **High-valence repair.** Process the current largest violation first. Try a legal interior edge flip that decreases its valence without making either opposite endpoint exceed eight. If that fails, remove the overloaded nonboundary node and triangulate its ordered one-ring. When plain ear clipping would regress local quality or exceed (L/h=1.55), partition the ring into balanced sectors and replace the center with two to eight distributed Steiner nodes. Candidate partitions are ordered by maximum predicted valence, count of new (L/h>1.55) elements, minimum quality, maximum (L/h), and added-node count. A removable non-hard boundary node may instead use the guarded boundary-fan branch.

Independent legal edge flips are selected from one topology scan and committed in disjoint batches. If a global gate rejects a batch, halve it until the conflicting edit is isolated. This changes the dominant cost from one full-mesh audit per candidate toward one audit per accepted batch.

Before sequential cavity work, identify legal flips for every violating node from one topology scan. Select non-overlapping two-triangle flip patches and apply up to 64 in one transaction. If a batch fails the global quality/target gate, backtrack through smaller batches; unbatched or newly exposed cases return to the sequential priority queue. This retains the hard transaction contract while avoiding a whole-mesh topology rebuild for every independent flip.

When adjacent violating nodes form a simple connected interior component, treat the union of their incident triangles as one zipper cavity. Require one ordered outer ring with no boundary/fixed node inside the removal set. Remove the cluster together and build a smaller balanced topology from sector nodes plus a central node; the delivered elements remain triangles. Prefer the fewest replacement nodes that satisfy valence and (L/h) gates, then favor equilateral quality. This operation may still add nodes when the outer ring has too many connections for a coarser patch; its primary purpose is connectivity redistribution, not unconditional coarsening. Reject the entire cluster transaction if the cavity is non-simple or any global gate regresses.

For an admissible open-boundary vertex removal, retriangulating its fan can transfer the overload to an adjacent boundary node. Treat removal plus local stabilization as one atomic boundary-strip transaction: rebuild the affected topology, repair any new local overload with a legal spoke flip or guarded cavity edit, and only then evaluate global invariants. The boundary node remains removable only when its chord-deviation and cumulative-area limits pass. This prevents the useful first edit from being rejected merely because the violation moved during an intermediate state.

Use `diagnose_high_valence.py` before and after clearing. Its inventory distinguishes `edge_flip`, `interior_cavity`, `guarded_boundary_cavity`, `hard_boundary_blocked`, and `unordered_interior_ring` cases. A supplied conditioning report promotes actual rejected node lineage to the first local-map panels, which prevents unattempted budget remainder from being misreported as algorithmic failure.

## Restricted spring equilibration

Every accepted topology edit is followed by up to three short relaxation cycles. Seed the changed cavity, expand only two graph rings, keep all boundary and out-of-patch nodes fixed, and run at most six damped spring iterations per cycle with damping 0.30, per-step cap (0.08h), and equilateral-shape weight 0.25. Rebuild the two-ring mask after every accepted topology change; do not accumulate a broad moving region.

The finite-rest-length energy is

\[
E(x)=\frac12\sum_{(i,j)\in E}
\left(\frac{L_{ij}-\ell^*_{ij}}{\ell^*_{ij}}\right)^2
+\frac{\beta}{2}\sum_T(1-q_T)^2,
\]

with the same target-aware rest-length construction used by `spring-relax-v1`.

## Transaction gates

The following detailed tolerances describe legacy/research edit ladders unless a
paragraph explicitly names `minimal-topology-v1` or `autonomous-thin-v1`. The
default paths always use the benchmark-first ordering above.

Snapshot before every edit. Restore it unless all of the following hold:

- every signed area stays positive above the scale-aware tolerance;
- protected chain edges and ordered OBC pairs remain present;
- connected-component count and nonmanifold-edge count do not regress;
- hard anchors survive unchanged and boundary edits obey their explicit envelopes;
- the targeted defect count or severity improves;
- controlled (q_{L3\sigma}), first-percentile quality/angle, and (L/h) tails do not regress beyond their documented tolerances;
- valence work reduces either the count of nodes above eight or their total excess. The default permits no new (L/h>1.55) triangles; an explicit, recorded integer budget may be supplied for a hard-gate closure transaction when the scientific tradeoff is accepted.

Wrap each high-valence stage and its immediate superthin cleanup in a second, outer transaction. The combined branch commits only when valence and target-quality non-regression pass, no new superthin severity or singly connected triangle remains, boundary component/degree audits do not regress, and every ordinary invariant above passes. Otherwise restore coordinates, connectivity, chains, lineage, targets, area accounting, and the edit ledger together.

The outer protocol repeats prune, superthin repair, and high-valence repair for at most four rounds, stopping early when no operation is accepted or when both the superthin count and valence violations reach zero. Write the terminal mesh, delivered boundary metadata, node lineage, and `mesh_edit_ledger.json` even when a remaining hard gate requires `needs_review`.
