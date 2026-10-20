# Scientific extent, priorities, and boundary placement

Begin with the requested dynamics. For `barotropic_tide`, ask which entrances, channels, islands and shelf connections control tidal propagation and depth-integrated transport. A `baroclinic` request additionally needs a reasoned view of stratification, vertical exchange and the broader water masses involved. Neither purpose automatically requires a maximal geographic domain.

Choose the smallest scientifically adequate region, with enough space for sound boundary placement. Additional upstream reaches, peripheral bays or deep offshore waters need a physical justification. An extent selected solely because a data catalog offers that box is not a scientific decision.

Essential features are those whose loss changes the requested process or disconnects a necessary water pathway. An optional feature can receive a coarse representation or be omitted when inspected evidence supports the compromise. Record the decision before interpreting a successful mask correction as acceptance. Future refinement interests describe where later nesting could improve fidelity; the current workflow does not generate or validate those nested grids.

Several hundred physical cells per direction is useful starting guidance for the downstream root grid, not a hard limit. A larger grid is valid with a scientific and computational-budget rationale. A small secondary feature should not force uniform fine spacing across the entire region. Consider supported smooth grading, purpose-consistent optional simplification, and later nesting while preserving all essential water connections.

Open boundaries should follow intentional exchange interfaces and avoid cutting real islands, leaving isolated slivers, or truncating essential entrance geometry. Multiple open sides are legitimate for a structured grid. Record mixed land/water sides honestly. A geographic four-sided polygon need not match the final numerical footprint exactly; the fit review must explain both boundary placement and added coverage.

Inspect the overview and each feature closeup. Check the shape of the domain as well as feature containment: a point can be inside while its passage is blocked, an island is cut, or a physically irrelevant area dominates cost. Compare any shoreline or bathymetry discrepancy against verified sources. A bridge deck appearing as elevated terrain may be a source artifact, but an actual island or solid causeway is physical land; preserve it. Any artifact correction belongs in an evidence-backed downstream mask/bathymetry decision. Bridge-pier drag and wetting/drying treatment are later modeling choices.

Record the physical reasons for accepting a plan and the limits of any simplification. Opening and actually inspecting the maps is mandatory at every revised stage. Automated containment, source hashing and rendered images support review; they do not perform it.
