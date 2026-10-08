# Grid request and delivery

Paths are relative to the request. A complete request may supply `region_request` (path to a region request) instead of `region_delivery`; `prepare` produces the region delivery and a resumable grid request. Review that region delivery before `fit`.

```json
{
  "region_delivery":"01_regions/region_delivery.json", "region_id":"outer",
  "parameters":{"spacing_m":500,"padding_m":2000,"hmin_m":2,"rx0_max":0.2,"N":40,"theta_s":6,"theta_b":2,"hc_m":20,"max_cells":1000000},
  "bathymetry":{"path":"sources/bathymetry.nc","variable":"elevation_m","positive":"up","manifest":"sources/metadata.json","vertical_datum":"Explicit source reference","warnings":[]},
  "protected_wet_features":[{"name":"inlet","point_lonlat":[-75.03,38.83],"max_snap_m":750,"review_radius_km":12}]
}
```

Optional `rotation_deg` is counterclockwise from east in the local projected plane; otherwise fit the minimum-area enclosing rectangle. Spacing applies in local conformal coordinates; actual geodesic spacing is diagnosed. `padding_m` extends scientific coverage before cell alignment.

`grid_fit.json` binds the request, region delivery/review, geometry parameters and map. `grid_delivery.json` binds `remora_grid.nc`, fit/review, sources, `mask_edits.json`, indexed `mask_changes.npz`, `validation.json`, `inputs.grid`, and maps. Reviews bind their exact manifest. Changed inputs require a new attempt.

NetCDF dimensions use `(eta,xi)` and one outer rho halo: physical `(nx,ny)` => rho `(ny+2,nx+2)`, u `(ny+2,nx+1)`, v `(ny+1,nx+2)`, psi `(ny+1,nx+1)`. The physical boundary lies on the outer psi lines. `remora.n_cell = nx ny N`. Coordinate metres are logical projected-grid x/y; geographic longitude/latitude are present at every stagger. Earth-geodesic metrics are independent of logical x/y increments.

Exported fields: x/y and lon/lat at rho/u/v/psi; h/hraw, pm/pn, angle/f; mask_rho/u/v/psi; stretching curves and parameters. CDF5 is written directly with netCDF4 (`NETCDF3_64BIT_DATA`). Vertical layer arrays are computed for validation/sections, not stored as unnecessarily large redundant grid fields.

All wet logical edges are reported individually. Production boundary conditions are later configuration work; no single FVCOM offshore side is assumed. Local reader receipts and private HPC account/path details belong in workspace evidence, not the distributable skill.
