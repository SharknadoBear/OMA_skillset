# Purpose-driven grid choices

Record `science.regime` and its physical `rationale` before choosing a grid. Keep the scientific prompt in `science_request.json` free of invented numerical prescriptions. The case agent chooses geometry, spacing, layer count and stretching for that request and records those choices in `grid_request.json`.

## Horizontal domain and resolution

Start from the reviewed region's essential water connections and boundary intents. Design the root grid to represent the requested dynamics with proportionate cost. Several hundred cells in each physical direction is soft guidance; a larger design needs both a scientific reason and a credible memory/runtime budget. Explicit `max_cells` remains a computational guard, not a scientific resolution criterion.

Keep essential channels and entrances connected, and retain islands that control the flow. Optional features may be simplified or excluded only with inspected source evidence and a physical rationale. A very small secondary feature does not justify fine uniform spacing throughout a large region. Consider smooth axis grading, an appropriate footprint, or later nesting; the current workflow generates only one root grid.

Avoid arbitrary upstream extensions, peripheral waters, and deep offshore area. Added coverage must support boundary placement or the requested dynamics. Open edges should align intentionally with exchange interfaces and avoid cutting islands or essential passages. Multiple open sides are valid. Explain how the geographic side intents map onto numerical `xlo`, `xhi`, `ylo`, and `yhi`; a mixed land/water edge is not automatically an error.

The supported conformal map permits moderate curvature or taper and smooth redistribution along the two coordinate axes. It is not a general shoreline-fitting mesh generator or arbitrary localized refinement. Use the quantitative Jacobian, orthogonality, neighbor-spacing and aspect-ratio diagnostics together with actual footprint and feature-map inspection. Grading along one logical axis affects a whole coordinate band; evaluate its effect elsewhere in the domain.

## Vertical purpose and bathymetry

For `barotropic_tide`, the case agent chooses 8–20 layers and explains the choice. Use rx0 ≤ 0.2 and actual layer-aware Haney rx1 < 9, preferably < 7, as the default design policy. Thresholds remain configurable, and any departure needs a scientific rationale. Do not inherit 40 layers merely because an old request used that number. Choose theta_s, theta_b, and hc with the depth range and intended vertical distribution in mind.

For `baroclinic`, choose layers and stretching to represent the relevant stratification and vertical exchange. A barotropic layer-count policy is not evidence of adequate baroclinic resolution. More layers alone do not establish physical fidelity, and fewer layers solely to suppress a diagnostic is not a scientific design.

Condition bathymetry in natural-log depth using the actual vertical configuration. Examine both rx0 and rx1, depth-change maps, sections, and volume/section-area changes. Passing roughness thresholds cannot justify an implausible entrance, altered sill, removed island or excessive change in the important depth range. Reconsider source data, extent, grading, stretching or optional-feature treatment when conditioning would damage the science.

An elevated bridge deck in a bathymetry/topography raster can create a false land barrier. Confirm that interpretation against reliable geographic evidence before preparing a corrected source artifact. Retain the original source and hashes, record what changed and why, and inspect the result. A genuine island, embankment or solid causeway remains physical land. Never turn unknown source coverage into water. Bridge-pier drag and wetting/drying behavior belong to a later model configuration.

## Acceptance

Actually inspect every region, fitted-footprint and final-grid map before recording its review. Check the overview and the important closeups; numerical connectivity does not prove a passage has enough cells or a defensible cross-section. Rendering an image, listing its path or passing a hash check does not constitute visual inspection.

Record unresolved compromises plainly. If the domain fails an essential connection, geometry criterion, source-coverage check, or configured roughness gate, revise the attempt. Keep local file validation, agent scientific acceptance, later human acceptance, and executable-reader evidence distinct. No grid-generation outcome alone demonstrates a stable or scientifically validated tidal simulation.
