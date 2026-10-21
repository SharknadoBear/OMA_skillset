# Grid request and delivery

Paths are relative to the request. A complete request may supply `region_request` (path to a region request) instead of `region_delivery`; `prepare` produces the region delivery and a resumable grid request. Review that region delivery before `fit`. The following illustrates the v2 scientific request; values are examples, not a prescribed design for another case.

```json
{
  "region_delivery":"01_regions/region_delivery.json",
  "region_id":"outer",
  "science":{
    "regime":"barotropic_tide",
    "rationale":"Represent depth-integrated tidal exchange through the essential entrance."
  },
  "design":{
    "coverage_rationale":"Include the essential water connections and a defensible shelf boundary.",
    "resolution_rationale":"Concentrate cells toward the entrance while grading smoothly toward the shelf.",
    "cell_budget_rationale":"The chosen dimensions support entrance resolution within the available memory budget.",
    "boundary_intents":{
      "xlo":{"intent":"mixed","rationale":"The coastal side intersects land and the intended water connection."},
      "xhi":{"intent":"open","rationale":"Shelf exchange boundary."},
      "ylo":{"intent":"open","rationale":"Alongshore exchange boundary."},
      "yhi":{"intent":"mixed","rationale":"Retain the northern coast and water exchange."}
    }
  },
  "geometry":{
    "kind":"conformal_graded",
    "nx":320,
    "ny":240,
    "xi_control":[[0,0],[0.4,0.25],[1,1]],
    "eta_control":[[0,0],[1,1]],
    "c2":[0.03,0],
    "c3":[0,0],
    "max_neighbor_ratio":1.15
  },
  "parameters":{
    "spacing_m":500,"padding_m":2000,"hmin_m":2,"rx0_max":0.2,
    "N":12,"theta_s":6,"theta_b":2,"hc_m":20,
    "vertical_rationale":"Twelve layers are chosen for this illustrative depth-integrated tidal design.",
    "rx1_max":9,"rx1_preferred":7,"rx1_target":7,"max_cells":1000000
  },
  "bathymetry":{"path":"sources/bathymetry.nc","variable":"elevation_m","positive":"up","manifest":"sources/metadata.json","vertical_datum":"Explicit source reference","warnings":[]},
  "protected_wet_features":[{"name":"main entrance","point_lonlat":[-75.05,38.85],"max_snap_m":750,"review_radius_km":12}]
}
```

`science.regime` is `barotropic_tide` or `baroclinic`. New requests with `science` or `geometry` require a regime rationale, the three design rationales above, an explicit integer `N`, and `parameters.vertical_rationale`. Barotropic requests require 8–20 layers. The case agent selects the actual values. `rx1_target` defaults to the preferred bound 7; `rx1_max` defaults to 9. Final maximum rx1 must be strictly below its hard bound; preferred attainment is reported separately. Legacy requests without `science` and `geometry` preserve the earlier uniform rotated grid and parameter defaults, including N=40; those defaults are compatibility behavior, not guidance for new tidal requests.

In the selected outer region, every essential water point must have a `protected_wet_features` entry with the exact same name and coordinates. Essential water boxes require at least one representative protected point with the same name inside the box; declare essential passages individually. A region feature point defaults to `feature_type: water`, a box to `coverage`; mark islands explicitly `land`. Essential land points must remain coastline land and dry in the final grid. Coverage boxes are extent requirements and do not turn the enclosed area into mandatory water. Child-region feature names do not impose root-grid anchors.

`parameters.rotation_deg`, when supplied, is counterclockwise from east in the local projected plane; otherwise the seed orientation comes from a minimum-area enclosing rectangle. `padding_m` extends scientific coverage. For new geometry, `nx` and `ny` are explicit physical cell counts; `spacing_m` seeds the projected enclosure and does not prescribe the final varying grid spacing. Actual spacing is `1/pm` and `1/pn`.

Each axis control is `[logical_fraction, position_fraction]`. Both columns must strictly increase from `[0,0]` to `[1,1]`. Smooth monotone interpolation redistributes the cells; identity controls retain uniform logical spacing. `c2` and `c3` are dimensionless `[real,imaginary]` coefficients of the bounded complex map described in [numerics](numerics.md). Zero coefficients retain the rotated rectangular footprint. Choose curvature and grading for the scientific geometry and document why identity choices are appropriate when used.

The fit automatically encloses the densified region with padding using bounded dilation. Inspect added coverage and cell size after fitting. `max_neighbor_ratio` defaults to 1.15 and must be in `(1,1.5]`; actual neighboring geodesic spacing ratios are checked in both directions. This is smooth coordinate redistribution, not arbitrary local refinement or numerical nesting.

`design.boundary_intents` requires all four logical sides, each with `intent` (`open`, `land`, `mixed`, or `undetermined`) and nonempty `rationale`. Translate geographic `side_index` intent after fitting rather than assuming that each original side becomes one particular logical edge. These records do not supply executable boundary-condition values.

## Evidence-backed source and mask adjustments

An optional region feature may be explicitly closed in the numerical mask using a top-level `mask_adjustments` list:

```json
{
  "mask_adjustments": [{
    "action": "close_optional",
    "feature_name": "secondary embayment",
    "vertices_lonlat": [[-75.6,39.2],[-75.4,39.2],[-75.4,39.4],[-75.6,39.4]],
    "rationale": "The reviewed optional recess is omitted at this root-grid scale.",
    "evidence": [{
      "source": "sources/geographic_assessment.json",
      "finding": "The retained essential transport path remains outside this optional recess."
    }]
  }]
}
```

`feature_name` must match a declared `optional` feature in the selected region. The polygon selects cells to dry; this mechanism never opens water or overrides real land. All essential protected wet anchors are checked after closure. Initial source-derived masks remain available for review. Inspect the changed cells, named feature and all essential connections before acceptance.

For an independently supported bathymetry/topography artifact, prepare a new source rather than opening cells through a mask override:

```text
python scripts/conditioning.py source-artifact-repair --primary PRIMARY.nc --variable elevation --primary-positive up --support SUPPORT.nc --support-variable elevation --support-positive up --selection-npz SELECTION.npz --support-reference "independent source citation" --rationale "specific evidence for selected artifacts" --output-dir NEW_DIR
```

`SELECTION.npz` contains binary `suspect_cells` and `mapped_land` arrays matching the source `(lat,lon)` shape. Both NetCDFs must have metre units, one-dimensional monotone lon/lat coordinates and identical coordinate arrays. `--primary-positive` and `--support-positive` specify `up` or `down`. The supporting file and hash must differ from the primary. The caller explicitly selects suspected artifacts; the helper does not discover bridge decks automatically.

Targets must be finite, outside mapped real land and have finite negative supporting elevations after sign conversion. The helper rejects missing coverage and inadequate reachable original wet donors. Optional `--radius-cells` defaults to 2 and `--min-neighbors` to 3. Original wet donors must be finite, negative and nonselected, and reachable without crossing mapped land. The selected samples are replaced using inverse squared index-distance interpolation; all other samples remain unchanged.

Outputs are `repaired_source.nc`, `source_artifact_repair.json` with hashes, rationale and donor records, and `source_artifact_repair.png` showing primary, repaired and supporting fields. The output directory must be new and outside installed skills. Open and inspect the map before downstream use; a repair helper's output is not scientific acceptance. Its `visual_review_status: pending` is the immutable generation-time state: retain a separate review receipt binding the repair-report/map hashes, or explicitly cite the actual repair-map inspection in the later grid review. Do not rewrite the repair manifest to mark it accepted. Retain original and supporting sources, and point `bathymetry.path`/`manifest` to the reviewed repaired source and its provenance. Preserve true islands and solid causeways. Bridge-pier drag and wetting/drying remain later modeling choices.

`grid_fit.json` binds the request, region delivery/review, geometry parameters and map. `grid_delivery.json` binds `remora_grid.nc`, fit/review, sources, `mask_edits.json`, indexed `mask_changes.npz`, `validation.json`, `inputs.grid`, and maps. Reviews bind their exact manifest. Changed inputs require a new attempt.

NetCDF dimensions use `(eta,xi)` and one outer rho halo: physical `(nx,ny)` => rho `(ny+2,nx+2)`, u `(ny+2,nx+1)`, v `(ny+1,nx+2)`, psi `(ny+1,nx+1)`. The physical boundary lies on the outer psi lines. `remora.n_cell = nx ny N`. Coordinate `x_*`/`y_*` metres remain logical coordinates for compatibility. New grids also carry physical `projected_x_*`/`projected_y_*`; geographic longitude/latitude are present at every stagger. Earth-geodesic metrics follow the same continuous physical mapping and must not be inferred from logical x/y increments.

Exported fields: logical x/y, lon/lat and new-grid physical projected x/y at rho/u/v/psi; h/hraw, pm/pn, angle/f; mask_rho/u/v/psi; stretching curves and parameters. CDF5 is written directly with netCDF4 (`NETCDF3_64BIT_DATA`). Vertical layer arrays are computed for validation/sections, not stored as unnecessarily large redundant grid fields.

All wet logical edges are reported individually. Production boundary conditions are later configuration work; no single FVCOM offshore side is assumed. Local reader receipts and private HPC account/path details belong in workspace evidence, not the distributable skill.
