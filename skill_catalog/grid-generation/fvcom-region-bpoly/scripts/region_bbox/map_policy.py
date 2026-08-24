from __future__ import annotations

from typing import Any

from .normalization import canonical_region_key


def domain_scale_for_request(request: dict[str, Any] | str, features_doc: dict[str, Any] | None = None) -> str:
    if features_doc and features_doc.get("domain_scale"):
        return str(features_doc["domain_scale"])
    key = canonical_region_key(request)
    if key == "murderkill":
        return "small_estuary"
    return "regional"


def resolve_basemap_provider(
    request: dict[str, Any] | str,
    requested_provider: str | None,
    features_doc: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve the user-facing basemap setting to a concrete plotting provider."""
    requested = (requested_provider or "auto").strip().lower()
    scale = domain_scale_for_request(request, features_doc)
    policy = {
        "schema_version": "map_detail_policy_v1",
        "requested_provider": requested,
        "domain_scale": scale,
        "resolved_provider": requested,
        "provider_chain": [],
        "target_zoom": None,
        "target_zoom_range": None,
        "reason": "explicit_provider",
    }
    if requested not in {"", "auto"}:
        if requested in {"road", "street", "road_detail"}:
            policy.update(
                {
                    "resolved_provider": "road_detail",
                    "provider_chain": ["Esri.WorldStreetMap", "CartoDB.Voyager", "OpenStreetMap.Mapnik"],
                    "target_zoom": 13 if scale == "small_estuary" else None,
                    "target_zoom_range": [13, 15] if scale == "small_estuary" else None,
                    "reason": "explicit_road_detail_provider",
                }
            )
            return "road_detail", policy
        if requested in {"topo", "topographic", "regional_context", "regional-context"}:
            policy.update(
                {
                    "resolved_provider": "topo",
                    "provider_chain": [
                        "Esri.WorldTopoMap",
                        "OpenTopoMap",
                        "CartoDB.Voyager",
                        "OpenStreetMap.Mapnik",
                    ],
                    "reason": "explicit_regional_topographic_provider_chain",
                }
            )
            return "topo", policy
        return requested, policy

    key = canonical_region_key(request)
    if scale == "small_estuary" or any(term in key for term in ["creek", "river"]):
        resolved = "road_detail"
        reason = "small_estuary_or_creek_scale_requires_high_detail_road_basemap"
        policy.update(
            {
                "provider_chain": ["Esri.WorldStreetMap", "CartoDB.Voyager", "OpenStreetMap.Mapnik"],
                "target_zoom": 13,
                "target_zoom_range": [13, 15],
            }
        )
    else:
        resolved = "topo"
        reason = "regional_or_lake_scale_uses_topographic_context"
        policy.update(
            {
                "provider_chain": [
                    "Esri.WorldTopoMap",
                    "OpenTopoMap",
                    "CartoDB.Voyager",
                    "OpenStreetMap.Mapnik",
                ]
            }
        )
    policy.update({"resolved_provider": resolved, "reason": reason})
    return resolved, policy


def side_focus_radius_km(request: dict[str, Any] | str, features_doc: dict[str, Any] | None = None) -> float:
    if domain_scale_for_request(request, features_doc) == "small_estuary":
        return 12.0
    return 45.0
