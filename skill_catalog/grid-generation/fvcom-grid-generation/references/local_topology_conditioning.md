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

1. **Redundant degree-3/4 vertex pruning.** An unfixed interior node with an ordered 3- or 4-node ring is removable only when a legal cavity triangulation has positive signed area, maximum (L/h\le1.25), minimum (q_T\ge0.40), minimum angle at least (28^\circ), and no resulting neighbor exceeds valence eight. For a quadrilateral, evaluate both diagonals and lexicographically favor smaller maximum (L/h), larger minimum quality, then larger minimum angle.

2. **Interior superthin-pair collapse.** If two superthin triangles share an unprotected interior edge (e=(a,b)), both endpoints are movable, the simplicial link condition holds, and (L_{ab}/h_{ab}\le0.50), merge (a) and (b) at

   \[
   x_{ab}=\tfrac12(x_a+x_b).
   \]

   Delete the two degenerate incident records created by the merge, compact the cavity, and retain source-node lineage in the edit ledger.

3. **Boundary superthin repair.** Examine the longest side of the worst boundary-adjacent superthin triangle. If that side is a one-sided protected arc whose endpoints have the same boundary kind, insert its midpoint on the arc and update the ordered chain and OBC nodestring. Otherwise, consider removing a non-hard boundary vertex and triangulating its local fan. Removal is accepted only when the vertex-to-neighbor chord deviation is below

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

The outer protocol repeats prune, superthin repair, and high-valence repair for at most four rounds, stopping early when no operation is accepted or when both the superthin count and valence violations reach zero. Write the terminal mesh, delivered boundary metadata, node lineage, and `mesh_edit_ledger.json` even when a remaining hard gate requires `needs_review`.
