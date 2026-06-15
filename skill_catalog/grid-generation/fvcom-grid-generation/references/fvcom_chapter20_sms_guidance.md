# FVCOM Manual v3.1.6 Chapter 20 Guidance

Use this reference when building or reviewing FVCOM grid-generation code and `.2dm` output.

## Workflow Interpreted for This Skill

Chapter 20 describes the SMS path used by the FVCOM team:

1. Prepare coastline and bathymetry.
2. Project geographic lon/lat to a local Cartesian system for coastal applications, while preserving enough information to return to lon/lat.
3. Open coastline data in SMS.
4. Create an initial open-boundary feature arc.
5. Smooth the open-boundary line by moving/adding feature vertices.
6. Redistribute feature vertices to set horizontal resolution.
7. Build polygons, convert map features to a 2D mesh, and save the mesh.
8. Unlock and inspect mesh quality.
9. Create/select the open-boundary nodestring, renumber it, and save the mesh.
10. Interpolate bathymetry to mesh nodes and save the mesh/scatter output.
11. Display the mesh and interpolated bathymetry for visual review.

The Python workflow should preserve these ideas even when it does not use the SMS GUI.

## Required Quality Checks

The manual recommends checking:

- minimum interior angle: 30 degrees;
- maximum interior angle: 130 degrees;
- maximum bathymetric slope: 0.1;
- element area change: 0.5;
- connecting elements: 8 or fewer, with 8 the maximum allowed for FVCOM.

This skill treats these as v1 acceptance gates unless the user explicitly relaxes them for exploratory meshes.

## Open Boundary Expectations

FVCOM can run without ideal open-boundary geometry, but the manual warns that numerical noise can increase if high-frequency waves reflect from the open boundary. It recommends making one interior edge of an open-boundary triangle normal to the open boundary.

The code therefore must:

- tag the open boundary explicitly;
- write an `NS` nodestring in the `.2dm` file;
- keep open-boundary nodes ordered along the boundary;
- compute a boundary-normality diagnostic for adjacent triangles;
- show the open boundary clearly in map diagnostics.

## Bathymetry Expectations

FVCOM requires positive water depth at wet points. The skill must normalize bathymetry to positive down and prevent NaN or nonpositive exported wet-node depths.
