# Grid delivery input contract

`fvcom_grid_delivery_contract_v1` is the complete, immutable handoff from grid
generation to FVCOM preconfiguration. Relative paths resolve from the contract
file. Hash ordered node arrays as compact JSON, such as `[1,2,3]`, with
SHA-256.

```json
{
  "schema_version": "fvcom_grid_delivery_contract_v1",
  "mesh": {
    "path": "fvcom_grid.2dm",
    "sha256": "...",
    "node_count": 56856,
    "triangle_count": 104010,
    "coordinate_crs": "EPSG:4326",
    "horizontal_crs": "EPSG:32615",
    "horizontal_units": "m"
  },
  "open_boundaries": [
    {
      "obc_id": "obc_001",
      "nodestring_id": 1,
      "node_ids": [1, 2, 3],
      "node_order_sha256": "...",
      "obc_type": "prescribed"
    }
  ],
  "excluded_nodestring_ids": [],
  "sigma": {"levels": 10, "type": "UNIFORM"}
}
```

`coordinate_crs` describes the x/y values serialized in the source 2DM;
`horizontal_crs` is the projected-metric CRS written to FVCOM DAT files.
They may be identical when a 2DM is already metric.

Every node list must exactly equal its SMS nodestring, including orientation
and order. Different OBCs may not share nodes. The source mesh may serialize
geographic longitude/latitude, but `horizontal_crs` and every output x/y value
must be projected meters; geographic degrees are never accepted as metric
FVCOM `_grd.dat` output.

An attempt override uses `fvcom_sponge_profile_v1`:

```json
{
  "schema_version": "fvcom_sponge_profile_v1",
  "attempt_id": "attempt_0002",
  "open_boundaries": {
    "obc_001": {"radius_multiplier": 2.0, "coefficient_multiplier": 0.5}
  }
}
```

Each rule may set `radius_m` or `radius_multiplier`, and `coefficient` or
`coefficient_multiplier`. Omitted OBCs retain their baseline. Unknown OBC IDs,
nonpositive controls, stale hashes, and attempt-ID mismatches fail closed.
