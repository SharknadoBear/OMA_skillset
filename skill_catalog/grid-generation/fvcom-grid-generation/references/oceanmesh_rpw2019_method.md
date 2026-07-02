# OceanMesh/RPW2019 Method Reference

Use this note when changing the pure-Python meshing or size-field implementation.

OceanMesh2D is retained as a method reference only. The local snapshot under `Resources/OceanMesh2D_Projection_snapshot/` is GPL-3.0. Do not copy or translate MATLAB source line-by-line into this skill.

Clean-room concepts to preserve:

- Represent the model domain as a constrained polygon with fixed outer and island boundary edges.
- Use a mesh-size function built from shoreline distance, feature scale, bathymetry depth, bathymetric slope/topographic length scale, gradation, and optional CFL limits.
- Use constrained Delaunay refinement and smoothing, with boundary nodes fixed.
- Keep boundary topology explicit: open boundary, land/model outer boundary, island boundaries, and interior wet-domain nodes.
- Preserve fixed boundary constraints before accepting the mesh.

For v1, the Python backend uses SciPy Delaunay, iterative boundary-midpoint insertion, circumcenter/edge midpoint refinement, and interior-node smoothing. This is OceanMesh-style, not a copied OceanMesh2D backend.
