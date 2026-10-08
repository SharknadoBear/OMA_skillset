# Region interface v1

UTF-8 JSON; paths resolve relative to the declaring request or manifest.

```json
{
  "name": "regional_case", "objective": "Resolve bay-shelf exchange",
  "coastline": {"path":"sources/land.geojson", "manifest":"sources/coastline_manifest.json", "bbox_wsen":[-76,38,-74,40]},
  "regions": [{
    "id":"outer", "parent_id":null, "role":"outer_domain",
    "purpose":"Include the bay and shelf connection",
    "vertices_lonlat":[[-75.7,38.6],[-74.5,38.6],[-74.5,39.6],[-75.7,39.6]],
    "required_features":[{"name":"bay entrance","point_lonlat":[-75.05,38.85],"review_radius_km":12}]
  }],
  "geographic_sources": []
}
```

Features accept `point_lonlat` or `bbox_wsen`; the whole feature must be covered. Four vertices are perimeter-ordered; a repeated closing vertex is accepted. Roles describe outer domains, nesting interests, or refinement interests. No AMR level is implied. Multi-root plans require an explicit `region_id` when gridding.

Outputs: `region_plan.json` (`remora_region_plan_v1`), GeoJSON polygons, maps, `region_delivery.json` (`remora_region_delivery_v1`), and a separate hash-bound review. Plan identity is independent of grid generation. Validation checks source, plan, map, and review hashes.

Portable deliveries include referenced sources or preserve their relative location. Do not place local private runtime information in the versioned skill.
