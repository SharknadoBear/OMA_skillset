# Region request and compatible v1 delivery

UTF-8 JSON; paths resolve relative to the declaring request or manifest. The v2 workflow adds optional scientific metadata while preserving `remora_region_plan_v1` and `remora_region_delivery_v1`. Existing requests without the new fields remain valid.

```json
{
  "name": "regional_case",
  "objective": "Resolve tidal exchange through the main entrance",
  "modeling_purpose": "barotropic_tide",
  "science_rationale": "The scientific target is depth-integrated tidal transport.",
  "coastline": {
    "path": "sources/land.geojson",
    "manifest": "sources/coastline_manifest.json",
    "bbox_wsen": [-76, 38, -74, 40]
  },
  "regions": [{
    "id": "outer",
    "parent_id": null,
    "role": "outer_domain",
    "purpose": "Include the bay entrance and its shelf connection",
    "vertices_lonlat": [[-75.7,38.6],[-74.5,38.6],[-74.5,39.6],[-75.7,39.6]],
    "features": [{
      "name": "main entrance",
      "point_lonlat": [-75.05,38.85],
      "priority": "essential",
      "feature_type": "water",
      "review_radius_km": 12
    }],
    "boundary_intents": [{
      "side_index": 1,
      "intent": "open",
      "rationale": "Connect the modeled entrance to shelf tidal exchange."
    }]
  }],
  "geographic_sources": []
}
```

`objective` and each region's `purpose` are scientific prose. Optional top-level `modeling_purpose` is `barotropic_tide` or `baroclinic`; `science_rationale`, when supplied, is nonempty prose. Both are retained in the plan. They do not choose numerical grid or vertical parameters.

Four vertices are perimeter-ordered; a repeated closing vertex is accepted. `side_index` 0 is vertex 0 to 1, then 1 to 2, 2 to 3, and 3 to 0. A `boundary_intents` entry has a unique side index, `intent` (`open`, `land`, `mixed`, or `undetermined`), and physical `rationale`. Several sides may be open. These are geographic planning records, not configured boundary conditions; after fitting, relate them explicitly to numerical `xlo`, `xhi`, `ylo`, and `yhi`.

`outer_domain` must have no parent. `nesting_interest` and `refinement_interest` require a parent. All parents must exist, relationships must be acyclic, and every child must lie inside its parent. Omitted and unknown legacy roles remain valid. A role never creates an AMR level. Multi-root plans require an explicit `region_id` when gridding.

## Features and decisions

Use `features` for new plans. Legacy `required_features` is still accepted; both lists may coexist, with unique feature names across them. A feature needs exactly one `point_lonlat` or `bbox_wsen`; `review_radius_km`, if supplied, must be finite and positive. The full shape of an included or simplified feature must be inside its region. A feature defaults to priority `essential`.

Optional `feature_type` is `water`, `land`, or `coverage`. A point defaults to `water`; a bounding box defaults to `coverage`. Mark island points explicitly `land` so they cannot be mistaken for wet connectivity anchors. A coverage box protects geographic extent, not an assertion that every enclosed cell is water.

| Priority | Disposition | Meaning |
| --- | --- | --- |
| `essential` | `included` (default) | Must be geographically covered and preserved through grid production. |
| `optional` | `included` (default), `simplified`, or `excluded` | A secondary feature can be simplified or omitted only with physical support. |
| `future_refinement` | `deferred` (default) | Records later nesting interest; makes no present-resolution promise. |

Every non-included disposition requires `rationale`. `simplified` and `excluded` also require at least one `evidence` entry with nonempty `source` and `finding`, for example:

```json
{
  "name": "secondary embayment",
  "point_lonlat": [-75.5, 39.3],
  "priority": "optional",
  "disposition": "simplified",
  "rationale": "Its detailed shoreline is secondary to the requested main-channel exchange.",
  "evidence": [{
    "source": "sources/geographic_assessment.json",
    "finding": "The assessed transport pathway follows the main entrance; the secondary recess is outside that pathway."
  }]
}
```

An excluded or deferred feature may lie outside the present region. Give closeups enough valid source coverage to support its review. Evidence references identify verified source URLs or retained workspace artifacts; the agent must inspect and substantiate them. The validator checks record structure, not the truth of the scientific argument. Preserve local evidence with the case; coastline source and manifest files are independently hash-bound by the delivery.

Feature metadata and boundary intents are retained in the plan and GeoJSON properties. Feature maps label priorities, and maps annotate declared side intents. For v2 numerical requests, every essential water point in the selected outer region requires a `protected_wet_features` point with the same name and coordinates. An essential water bounding box requires at least one protected representative point with the same name inside the box; declare important passages separately rather than assuming one point protects every connection. Essential land points are checked against coastline land and the final dry mask. These obligations apply to the selected region, not unrelated child-region feature names.

## Outputs and validation

Outputs are `region_plan.json`, GeoJSON polygons, maps, `region_delivery.json`, and a separate hash-bound review. Plan identity is independent of numerical grid generation. Validation checks original request, source, source manifest when supplied, plan, polygon, and map hashes, along with purpose, priorities, role graph and boundary-intent structure. An accepted review binds the exact delivery.

Portable deliveries include referenced sources or preserve their relative locations. Do not place local private runtime information in the versioned skill.
