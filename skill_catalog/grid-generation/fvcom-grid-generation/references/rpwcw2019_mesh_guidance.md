# RPWCW2019 Mesh-Design Guidance

Use this reference when changing mesh-size functions, bathymetry smoothing, or gradation controls.

## Core Principle

Roberts, Pringle, Westerink, Contreras, and Wirasaet (2019) argue for automatic, a priori unstructured mesh design based on shoreline geometry and seabed topography. The key practical lesson is to place fine resolution where the physical system demands it, while relaxing resolution elsewhere.

## Mesh-Size Functions to Preserve

The paper evaluates several interacting controls:

- shoreline distance/minimum size;
- feature-size refinement for narrow shoreline geometry;
- gradation limits on element-size transitions;
- topographic-length-scale refinement over steep seabed gradients;
- estuarine/submarine channel refinement.

The v1 Python skill implements the first, third, and fourth ideas directly. Feature-size and channel-specific refinement should remain explicit future extension points.

## Defaults for This Skill

- Use a conservative gradation default of 15 percent for stable, smooth transitions.
- Treat 35 percent gradation as experimental unless additional slope/channel constraints are active and the quality report is reviewed.
- Use depth caps to avoid overly coarse nearshore and shelf meshes.
- Use the topographic-length-scale form proportional to `depth / abs(grad(depth))`, scaled by `2*pi / slope_elements`, with `slope_elements` defaulting to 20.
- Apply slope refinement away from very shallow water to avoid excessive response to noisy nearshore bathymetry.

## Interpretation for FVCOM

For FVCOM preprocessing, the paper should guide where resolution is placed, while the FVCOM manual governs acceptance checks and `.2dm` boundary handling. A mesh should not be accepted solely because it is visually attractive; it must also pass angle, slope, area-change, connectivity, and open-boundary diagnostics.
