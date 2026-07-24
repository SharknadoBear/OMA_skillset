# OceanMesh/RPW Standalone Post-Generation Method Reference

Use this reference before changing standalone postprocessing or quality defaults. OceanMesh2D is GPL-3.0; preserve workflow concepts and independently implement Python algorithms rather than copying MATLAB code.

Normal `run_fvcom_grid.py` execution ends at the generation-time smoothed mesh and requires `postprocess_profile=none`. Run these profiles only through `postprocess_fvcom_mesh.py` for an explicit experiment, then compare against the untouched mesh before accepting the result.

## Sources and Fidelity

- RPW2019 Sect. 3.1.3 defines the scientific sequence: fix consistency, make boundaries traversable, remove singly connected elements, bound connectivity, and apply one direct implicit smoothing operation.
- The local Projection snapshot at commit `754d69a629d7b326383665123e7ea879d9db7040` adds poor-boundary deletion, thin-triangle collapse, categorical cleaner profiles, hill-climbing smoothing, and recursive cleanup.
- These sources differ. Keep `rpw2019` and `projection-medium` as distinct, named profiles rather than blending their defaults silently.

## Quality Mathematics

For triangle area \(A_E\) and edge lengths \(\lambda_i\), use

\[
q_E = \frac{4\sqrt{3}A_E}{\lambda_1^2+\lambda_2^2+\lambda_3^2},
\qquad
q_{L3\sigma}=\overline{q_E}-3\sigma_{q_E}.
\]

RPW2019 uses \(q_{L3\sigma}>0.75\) as a mesh-generation termination target. Postprocessing must also report the minimum and lower-tail quantiles because protected shoreline elements can dominate the worst cases.

Define node valence as the number of unique neighboring vertices, not the number of incident triangles.

## Profile Contracts

### `rpw2019`

- Disjoint-component area fraction: `0.25`.
- Singly connected processing: exhaustive.
- Default connectivity target: `6` unique neighbors.
- Smoothing: one sparse direct implicit solve with all protected boundary nodes fixed.
- Recursion: none; execute the documented sequence once.

### `projection-medium`

- Poor-boundary and thin-triangle cutoff: `q_E=0.25`.
- Effective meshgen disjoint fraction: `0.25`.
- Singly connected deletion: disabled by the medium profile.
- Default connectivity target: `8` unique neighbors.
- Smoothing: direct implicit when fixed points are supplied.
- Repeat while topology changes and `q_min<0.25`; stop on target, no change/protected stall, or eight passes.

## Protected-Boundary Adaptation

With `protect-all`, never delete, move, or collapse open, land, island, or frame boundary nodes and edges. Replace destructive boundary operations with quality-improving flips or one-ring cavity retriangulation that retains every constraint. Retain and classify a defect when no legal repair exists. Boundary protection takes precedence over literal destructive OceanMesh cleanup.

Require transactional operations: apply to a candidate state, audit positive area, manifold topology, all constraint edges, OBC ordering, and quality, then commit or roll back.

## During-Generation Context

OceanMesh periodically bisects edges with roughly `L/h > 2`, deletes edges with roughly `L/h < 1/2`, removes low-valence interior vertices, and removes extreme-angle triangles. The Python generator remains a constrained-Delaunay clean-room backend; postprocessing does not make it numerically identical to DistMesh.

## Required Diagnostics

Report before and after:

- `q_min`, `q_mean`, `q_std`, `q_L3sigma`, and quality quantiles;
- minimum-angle quantiles and counts below 20 and 30 degrees;
- `L/h` quantiles;
- true valence and counts above 6 and 8;
- adjacent-area change and bathymetric slope;
- connected components, boundary-degree anomalies, singly connected triangles, non-manifold edges, and nonpositive areas;
- protected-edge presence and ordered OBC pair integrity.
