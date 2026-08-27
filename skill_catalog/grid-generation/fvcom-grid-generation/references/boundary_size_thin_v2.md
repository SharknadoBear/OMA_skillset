# Adaptive Coastal v2 Boundary–Size–Thin Contract

Use this reference with `adaptive-coastal-v2`, the sole active boundary
resolution implementation. Archived legacy/v1 products are read-only evidence,
not generation paths.

## Contents

- [Prevention before repair](#prevention-before-repair)
- [Mandatory anchors](#mandatory-anchors)
- [Size-field priority](#size-field-priority)
- [Compound local repair](#compound-local-repair)
- [Acceptance gates](#acceptance-gates)

## Prevention before repair

Thin-triangle control starts with a semantically sampled boundary.  Boundary
representation spacing `h_repr` and interior dynamical spacing `h_dyn` are
related but distinct.  Fine source shoreline geometry may require boundary
vertices without forcing the same resolution through the full interior.

At a land/open-boundary junction, use one shared target `h_J` and blend along
the land chain with

```text
h_land(s) = max(h_land_base, h_J - g s),  g <= 0.15.
```

The junction floor may relax only soft shoreline/depth refiners. Mission
constraints and protected-passage representation remain hard and may override
it. CFL is diagnostic and does not set the production size target.

For opposing banks separated by wet width `W`, use the paired target

```text
h_pair = min(h_left, h_right, W / N_cross).
```

Use at least four elements across protected channels.  An unprotected passage
that cannot meet its configured cross-channel count at the permitted minimum
spacing is an upstream topology decision, never an arbitrary triangle deletion.

## Mandatory anchors

Both OBC landfalls are hard anchors. Stable sharp turns, spit tips, mission
features, and protected-passage control points are also anchors. Detect them from
multi-scale turn/curvature and chord-error evidence, then sample each
anchor-to-anchor interval without a short terminal remainder.  Preserve anchor
identity and coordinates through mesh serialization.

## Size-field priority

Use segment-interpolated boundary targets rather than nearest-vertex targets:

```text
h_b(x) = min_e((1-t) h_i + t h_j + g distance(x,e)).
```

Use the delivered solid-boundary targets directly and construct

```text
h_N = gradate_wet(min(h_solid, h_gradient, h_hydraulic))
```

Here `h_hydraulic` comes from the solid-only paired-bank medial skeleton, not an
imported drainage or thalweg line. The skeleton width and its continuous
importance-dependent element count control the medial target; a quintic
log-space transverse blend joins that target to `h_solid` at the banks.

For open domains, transfer from the propagated OBC target to `h_N` along
wet-domain distance:

```text
h_final = gradate_wet(exp((1-P(xi)) log(h_open) + P(xi) log(h_N)))
```

Use `xi=clip((d_wet-L_hold)/L_transition,0,1)` and
`P(xi)=6xi^5-15xi^4+10xi^3`. Apply the metric-aware eight-neighbor limiter only
over wet cells and record source attribution, hydraulic skeleton/corridor
metrics, propagated OBC target and wet distance, transition feasibility, and
gradation reduction. No out-of-coverage query may silently receive the domain
maximum.

## Compound local repair

Treat valence and extreme-shape repair as one transaction:

```text
snapshot
-> valence edit
-> identify newly created superthin component
-> least-invasive component repair
-> local valence stabilization
-> local and global audit
-> accept or restore the snapshot
```

The component candidate ladder is legal flip, local cavity retriangulation,
redundant exterior-ear removal, guarded vertex-to-source-arc weld, guarded
boundary-fan edit, then eligible split/collapse.  Under-resolved passage closure
is not part of this mesh-level ladder.

A redundant ear may be removed only if its protected sides remain represented,
the exterior remains traversable, no required connectivity changes, and actual
signed domain-area change stays within budget.  A weld must project to the
interior of its original source arc, remain outside protected junction/channel
buffers, satisfy displacement and altitude limits, resample the Eulerian target,
and preserve one-sided chains, anchors, manifold topology, and valence eight.

## Acceptance gates

Before triangulation require:

- both OBC landfalls and every mandatory anchor present;
- boundary `L/h <= 1.55`;
- spacing gradation `<= 0.15`;
- paired/kind-transition target ratios within the recorded contract;
- explicit retain/close/review decisions for detected passages;
- a valid resolved wet polygon and ordered metadata-aligned chains.

For the delivered mesh require:

- zero superthin elements (`q < 0.10` or minimum angle below 5 degrees);
- lower three-sigma equilateral quality `q_l3_sigma > 0.75`;
- unique-neighbor valence `<= 8`;
- positive signed areas, one manifold component, traversable exterior, zero
  singly connected elements, complete constraints, and ordered OBC nodes;
- adaptive OBC `p95(L/h) <= 1.55` and maximum `L/h <= 2`;
- exact hard-anchor survival and successful 12-decimal SMS roundtrip.

Existing full FVCOM angle, bathymetric-slope, and adjacent-area gates remain
authoritative.  A topology change that affects a scientifically meaningful wet
passage remains `needs_review` even when geometric gates pass.
