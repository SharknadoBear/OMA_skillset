# Aggressive Local Topology Conditioning

Use this note for `aggressive-local-v2`, `condition_mesh_local.py`, `repair_high_valence.py`, and `prune_redundant_vertices.py`.

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

The reusable V5 interaction engine moves fixed-connectivity vertices under edge, equilateral-angle, and positive-area barrier forces. Keep all boundary and hard-anchor nodes exact during the burst. Stop on the configured unified superthin trigger, recurring repaired lineage, nonpositive geometry, three rejected steps, or two checkpoint gains below `1e-5`. Select the highest-`q_l3_sigma` checkpoint within the trigger, run exhaustive closure, and roll back the complete cycle on any failed gate. Always deliver the last accepted zero-debt champion with no subsequent relaxation. Default limits are six committed cycles and 1,000 cumulative iterations. V5 is opt-in; do not promote `auto` unless both frozen Delaware baselines reach zero debt under identical defaults.

Treat wall-clock checks as safety gates, not advisory logging. Check before every relaxation iteration, component, candidate, audit, and serialization. Reserve enough time for one full-mesh audit; if an existing helper contains a non-preemptible audit that exceeds the reserve, record a hard-stop defect and do not begin policy screening from a nonzero or unreported closure.

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
