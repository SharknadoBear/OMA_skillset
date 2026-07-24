# Adaptive Coastal v2 Boundary–Size–Thin Contract

Use this reference with `adaptive-coastal-v2`.  The profile is opt-in; legacy
and `adaptive-coastal-v1` behavior remain compatibility paths.

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

The junction floor may relax only soft shoreline/depth refiners.  Mission,
retained-channel, and CFL constraints remain hard and may override it.

For opposing banks separated by wet width `W`, use the paired target

```text
h_pair = min(h_left, h_right, W / N_cross).
```

Use at least four elements across protected channels.  An unprotected passage
that cannot meet its configured cross-channel count at the permitted minimum
spacing is an upstream topology decision, never an arbitrary triangle deletion.

## Mandatory anchors

Both OBC landfalls are hard anchors.  Stable sharp turns, spit tips, mission
features, and channel-control points are also anchors.  Detect them from
multi-scale turn/curvature and chord-error evidence, then sample each
anchor-to-anchor interval without a short terminal remainder.  Preserve anchor
identity and coordinates through mesh serialization.

## Size-field priority

Use segment-interpolated boundary targets rather than nearest-vertex targets:

```text
h_b(x) = min_e((1-t) h_i + t h_j + g distance(x,e)).
```

Separate candidates into

```text
h_soft = min(h_boundary, h_depth, h_slope)
h_hard = min(h_mission, h_channel, h_CFL)
```

and inside a junction mask apply

```text
h_raw = min(h_hard, max(h_soft, h_junction_floor)).
```

Apply a metric-aware eight-neighbor gradation limiter and record the dominant
source, coverage, junction, channel, and gradation-reduction arrays.  No
out-of-coverage query may silently receive the domain maximum.

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
