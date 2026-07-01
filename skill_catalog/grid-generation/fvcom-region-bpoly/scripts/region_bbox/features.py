from __future__ import annotations

from typing import Any

from .geometry import RegionBPoly
from .normalization import canonical_region_key, normalize_request_text, request_text
from .scoring import ingredient_points


def cook_inlet_domain_variant(request: dict[str, Any] | str) -> str | None:
    if canonical_region_key(request) != "cook_inlet":
        return None
    text = normalize_request_text(request)
    wave_terms = [
        "wave-current",
        "wave current",
        "wave",
        "swan",
        "wave climate",
        "offshore wave",
        "wave forcing",
        "fetch",
    ]
    if any(term in text for term in wave_terms):
        return "cook_inlet_wave_fetch"
    return "cook_inlet_tidal_mouth"


def _bbox_feature(
    features: list[dict[str, Any]],
    feature_id: str,
    label: str,
    role: str,
    category: str,
    bbox: list[float],
    required: bool = True,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    record = {
        "id": feature_id,
        "label": label,
        "role": role,
        "category": category,
        "type": "bbox",
        "geometry": bbox,
        "required": required,
        "notes": notes,
    }
    if extra:
        record.update(extra)
    features.append(record)


def _considerations(text: str) -> dict[str, list[str]]:
    notes: dict[str, list[str]] = {
        "forcing_data_source_consideration": [],
        "offshore_extension": [],
        "upstream_river_extent": [],
        "channel_connectivity": [],
        "lake_connecting_channels": [],
        "island_archipelago_completeness": [],
        "geopolitical_separation": [],
        "offshore_boundary_exclusion": [],
    }
    if any(k in text for k in ["tide", "tidal", "offshore", "wave", "swan", "surge", "otec"]):
        notes["forcing_data_source_consideration"].append("Include an offshore forcing apron large enough for tidal/wave/open-ocean boundary data.")
        notes["offshore_extension"].append("Keep the ocean-facing side away from constricted inlets and major island barriers.")
    if any(k in text for k in ["river", "freshwater", "discharge", "salinity", "salt intrusion", "nutrient"]):
        notes["upstream_river_extent"].append("Include practical upstream gates for river inputs rather than ending at the river mouth.")
    if any(k in text for k in ["channel", "connect", "exchange", "residual", "flushing", "tidal energy"]):
        notes["channel_connectivity"].append("Do not cut mission-critical connected channels or exchange pathways.")
    if "lake " in text or any(k in text for k in ["lake_", "lake superior", "lake huron", "lake michigan", "lake ontario", "lake erie"]):
        notes["lake_connecting_channels"].append("Treat lake domains as closed ocean-boundary cases while preserving inlets, outlets, and connecting straits when mission-relevant.")
    if any(k in text for k in ["island", "archipelago", "aleut", "hawaii", "kodiak"]):
        notes["island_archipelago_completeness"].append("Include the complete island/archipelago context requested by the mission and avoid splitting the open boundary with islands.")
        notes["offshore_boundary_exclusion"].append("Check that the selected offshore side corridor is not cut by nearby islands outside the intended domain.")
    if any(k in text for k in ["salish", "southeast alaska", "se alaska", "haida", "strait of georgia", "boundary pass"]):
        notes["geopolitical_separation"].append("Review US/Canada boundary context so political separation does not clip hydrodynamic connectivity.")
    return notes


def infer_target_region_features(request: dict[str, Any] | str) -> dict[str, Any]:
    """Infer feature boxes that the four-sided bpoly must encompass.

    The output intentionally mirrors the old RegionBox ingredient idea while
    making the target-region features first-class artifacts for visual review.
    """
    if isinstance(request, dict) and isinstance(request.get("target_region_features"), dict):
        return request["target_region_features"]

    text = normalize_request_text(request)
    key = canonical_region_key(request)
    features: list[dict[str, Any]] = []

    if key == "puget_salish":
        broad = any(k in text for k in ["tidal energy", "all tidal", "all the tidal", "connect", "salish"])
        _bbox_feature(features, "puget_sound", "Puget Sound", "target_water_body", "target_region", [-123.35, 46.95, -122.15, 48.55])
        _bbox_feature(features, "hood_canal", "Hood Canal", "tidal_channel", "channel_connectivity", [-123.35, 47.25, -122.7, 48.15])
        _bbox_feature(features, "strait_juan_de_fuca", "Strait of Juan de Fuca", "offshore_forcing_corridor", "forcing_data_source_consideration", [-125.1, 48.0, -122.8, 48.6])
        if broad:
            _bbox_feature(features, "san_juan_islands", "San Juan Islands", "mission_critical_connected_tidal_channels", "channel_connectivity", [-123.35, 48.35, -122.65, 48.9])
            _bbox_feature(features, "haro_boundary_pass", "Haro Strait and Boundary Pass", "mission_critical_connected_tidal_channels", "channel_connectivity", [-123.35, 48.45, -122.75, 49.15])
            _bbox_feature(features, "strait_georgia", "Strait of Georgia", "mission_critical_connected_tidal_channels", "geopolitical_separation", [-124.0, 48.8, -122.5, 50.35])
    elif key == "long_island_sound":
        _bbox_feature(features, "lis_core", "Long Island Sound core", "target_water_body", "target_region", [-73.95, 40.75, -71.83, 41.37])
        _bbox_feature(features, "ny_harbor", "New York Harbor and East River", "connected_waterbody", "channel_connectivity", [-74.35, 40.35, -73.72, 40.85])
        _bbox_feature(features, "raritan_newark", "Raritan/Newark/Jamaica Bay context", "connected_waterbody", "channel_connectivity", [-74.35, 40.35, -73.7, 40.78])
        _bbox_feature(features, "race_block_island", "The Race and Block Island Sound", "eastern_open_ocean_context", "forcing_data_source_consideration", [-72.35, 40.95, -71.45, 41.35])
        _bbox_feature(features, "narragansett", "Narragansett Bay/Providence context", "connected_waterbody", "channel_connectivity", [-71.55, 41.25, -71.05, 41.85])
    elif key == "murderkill":
        _bbox_feature(features, "murderkill_estuary_core", "Murderkill estuary core", "target_estuary", "target_region", [-75.50, 39.02, -75.36, 39.16])
        _bbox_feature(features, "murderkill_upstream_tidal_creek", "Murderkill upstream tidal-creek gate", "river_input_context", "upstream_river_extent", [-75.61, 39.10, -75.45, 39.22])
        _bbox_feature(features, "murderkill_bay_mouth", "Immediate Murderkill/Delaware Bay mouth", "open_bay_context", "channel_connectivity", [-75.43, 38.96, -75.18, 39.12])
    elif key == "delaware":
        _bbox_feature(features, "delaware_estuary", "Delaware River estuary", "target_water_body", "target_region", [-75.8, 38.8, -74.8, 40.2])
        _bbox_feature(features, "upstream_nontidal", "Upstream/nontidal river context", "river_input_context", "upstream_river_extent", [-75.3, 39.7, -74.6, 40.35])
        _bbox_feature(features, "offshore_shelf", "Offshore tidal forcing apron", "offshore_buffer", "offshore_extension", [-75.5, 37.6, -73.5, 39.0])
    elif key == "chesapeake":
        _bbox_feature(features, "chesapeake_bay", "Chesapeake Bay", "target_water_body", "target_region", [-77.6, 36.7, -75.3, 39.7])
        _bbox_feature(features, "susquehanna_gate", "Susquehanna upstream input gate", "river_input_context", "upstream_river_extent", [-76.9, 39.35, -76.0, 39.95])
        _bbox_feature(features, "offshore_midatlantic", "Atlantic forcing apron", "offshore_buffer", "offshore_extension", [-76.2, 36.2, -74.2, 38.5])
    elif key == "aleutian":
        _bbox_feature(features, "aleutian_chain", "Aleutian Islands chain context", "required_geospatial_extent", "island_archipelago_completeness", [172.0, 51.0, -158.0, 55.5])
        _bbox_feature(features, "bering_sea_gate", "Bering Sea forcing side", "offshore_forcing_corridor", "forcing_data_source_consideration", [172.0, 54.0, -160.0, 57.5])
        _bbox_feature(features, "north_pacific_gate", "North Pacific forcing side", "offshore_forcing_corridor", "offshore_extension", [172.0, 48.8, -160.0, 52.2])
    elif key == "hawaii_state":
        _bbox_feature(features, "hawaiian_chain", "Hawaiian Islands chain", "island_chain", "island_archipelago_completeness", [-161.0, 18.5, -154.5, 23.0])
        _bbox_feature(features, "hawaii_ocean_apron", "Open-ocean OTEC apron", "offshore_buffer", "offshore_extension", [-161.8, 17.7, -153.8, 23.8])
    elif key == "hawaii_island":
        _bbox_feature(features, "hawaii_big_island", "Hawaii Island / Big Island", "target_island", "target_region", [-156.15, 18.75, -154.65, 20.30])
        _bbox_feature(features, "big_island_clean_ocean_apron", "Big Island clean offshore apron avoiding neighboring islands", "offshore_buffer", "offshore_extension", [-156.45, 18.15, -154.05, 20.36])
        _bbox_feature(
            features,
            "maui_nui_neighbor_islands_obstruction",
            "Maui Nui neighboring-island obstruction guard",
            "offshore_boundary_exclusion",
            "obstruction_guard",
            [-157.30, 20.42, -155.60, 21.35],
            required=False,
            notes="Big-Island-only domains must not place the offshore side through Maui Nui or neighboring islands.",
            extra={"guard_distance_km": 70.0, "blocks_final_pass": True},
        )
    elif key == "lake_superior":
        _bbox_feature(features, "lake_superior", "Lake Superior", "lake", "target_region", [-92.4, 46.2, -84.2, 49.2])
        _bbox_feature(features, "st_marys_outlet", "St. Marys River outlet context", "lake_outlet", "lake_connecting_channels", [-84.8, 46.1, -83.9, 46.7])
    elif key == "lake_ontario":
        _bbox_feature(features, "lake_ontario", "Lake Ontario", "lake", "target_region", [-80.2, 43.0, -76.0, 44.6])
        _bbox_feature(features, "st_lawrence_outlet", "St. Lawrence outlet context", "lake_outlet", "lake_connecting_channels", [-76.6, 44.0, -75.6, 44.6])
    elif key == "lake_erie":
        _bbox_feature(features, "lake_erie", "Lake Erie", "lake", "target_region", [-83.6, 41.2, -78.5, 42.9])
        _bbox_feature(features, "detroit_niagara_connection", "Detroit/Niagara connection context", "lake_connection", "lake_connecting_channels", [-83.5, 41.2, -78.8, 43.1])
    elif key == "cook_inlet":
        cook_variant = cook_inlet_domain_variant(request)
        if cook_variant == "cook_inlet_wave_fetch":
            _bbox_feature(features, "cook_inlet_full", "Cook Inlet full west/east estuary context", "target_estuary", "target_region", [-154.05, 58.80, -149.0, 61.55])
            _bbox_feature(features, "upper_cook_inlet", "Upper Cook Inlet river-input context", "river_input_context", "upstream_river_extent", [-151.7, 60.3, -149.0, 61.6])
            _bbox_feature(features, "ursus_cove_kamishak", "Ursus Cove / Kamishak west-side Cook Inlet context", "west_side_inlet_context", "target_region", [-154.05, 59.35, -153.35, 59.70])
            _bbox_feature(features, "augustine_island", "Augustine Island wave-generation context", "island_context", "island_archipelago_completeness", [-153.65, 59.22, -153.22, 59.50])
            _bbox_feature(features, "kodiak_island_context", "Kodiak Island included wave-fetch context", "island_context", "island_archipelago_completeness", [-154.9, 56.7, -151.0, 58.9])
            _bbox_feature(features, "cook_inlet_broad_wave_apron", "Broad Gulf of Alaska wave-fetch apron", "offshore_buffer", "offshore_extension", [-156.5, 55.8, -147.8, 58.8])
        else:
            _bbox_feature(features, "cook_inlet", "Cook Inlet mouth-gate domain", "target_estuary", "target_region", [-153.3, 58.95, -149.0, 61.5])
            _bbox_feature(features, "gulf_of_alaska_open_gate", "Gulf of Alaska forcing apron north/east of Kodiak", "offshore_buffer", "offshore_extension", [-151.8, 58.95, -148.7, 59.3])
            _bbox_feature(features, "upper_cook_inlet", "Upper Cook Inlet river-input context", "river_input_context", "upstream_river_extent", [-151.7, 60.3, -149.0, 61.6])
            _bbox_feature(
                features,
                "kodiak_island_obstruction",
                "Kodiak Island offshore-boundary obstruction guard",
                "offshore_boundary_exclusion",
                "obstruction_guard",
                [-154.9, 56.8, -151.7, 58.80],
                required=False,
                notes="Cook Inlet tidal-mouth domains should not route the offshore side across Kodiak Island.",
                extra={"guard_distance_km": 80.0, "blocks_final_pass": True},
            )
    elif key == "southeast_alaska":
        _bbox_feature(features, "se_alaska", "Southeast Alaska tidal channels", "target_region", "island_archipelago_completeness", [-139.8, 54.0, -128.0, 60.0])
        _bbox_feature(features, "haida_gwaii_context", "Haida Gwaii context", "boundary_split_guard", "geopolitical_separation", [-133.5, 51.5, -130.0, 54.5])
        _bbox_feature(features, "gulf_alaska_forcing", "Gulf of Alaska open-boundary apron", "offshore_buffer", "offshore_extension", [-140.2, 51.6, -128.0, 56.0])
    elif key == "columbia":
        _bbox_feature(features, "columbia_estuary", "Columbia River estuary", "target_estuary", "target_region", [-124.3, 45.8, -123.4, 46.45])
        _bbox_feature(features, "offshore_columbia", "Pacific forcing apron", "offshore_buffer", "offshore_extension", [-125.0, 45.5, -123.8, 46.7])
    elif key == "hudson":
        _bbox_feature(features, "hudson_estuary", "Hudson River estuary", "target_estuary", "target_region", [-74.3, 40.5, -73.6, 42.2])
        _bbox_feature(features, "ny_harbor", "New York Harbor / East River complexity", "needs_review_context", "channel_connectivity", [-74.3, 40.45, -73.7, 40.9])
    elif key == "san_francisco":
        _bbox_feature(features, "sf_bay", "San Francisco Bay and Delta context", "target_estuary", "target_region", [-123.0, 37.0, -121.3, 38.4])
        _bbox_feature(features, "pacific_gate", "Pacific forcing apron", "offshore_buffer", "offshore_extension", [-123.3, 37.4, -122.3, 38.1])

    domain_scale = "small_estuary" if key == "murderkill" else "regional"
    domain_variant = cook_inlet_domain_variant(request) if key == "cook_inlet" else None
    return {
        "schema_version": "target_region_features_v1",
        "source": "heuristic_prompt_decomposition",
        "request_text": request_text(request),
        "domain_scale": domain_scale,
        "domain_variant": domain_variant,
        "considerations": _considerations(text),
        "features": features,
    }


def features_as_ingredients(features_doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(feature) for feature in features_doc.get("features", [])]


def _wrap_lon(lon: float) -> float:
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def _unwrap_lon(lon: float, origin: float) -> float:
    x = lon
    while x - origin > 180:
        x -= 360
    while origin - x > 180:
        x += 360
    return x


def fit_bpoly_to_feature_boxes(
    bpoly: RegionBPoly,
    ingredients: list[dict[str, Any]],
    padding_fraction: float = 0.05,
    include_candidate: bool = False,
) -> tuple[RegionBPoly, str | None]:
    """Return a padded four-sided envelope that contains required feature boxes.

    This is intentionally a simple fallback refit for the feature-first workflow:
    specialized deformations get the first chance, then missed required boxes
    cause a four-corner envelope expansion around the features. The candidate
    is optional so a bad initial guess cannot contaminate the repaired extent.
    """
    required = [item for item in ingredients if item.get("required", True)]
    if not required:
        return bpoly, None

    raw_lons = [float(lon) for item in required for lon, _lat in ingredient_points(item)]
    origin = raw_lons[0] if raw_lons else bpoly.center_lon
    lons: list[float] = []
    lats: list[float] = []
    if include_candidate:
        lons.extend(_unwrap_lon(p[0], origin) for p in bpoly.polygon_lonlat()[:-1])
        lats.extend(p[1] for p in bpoly.polygon_lonlat()[:-1])
    for item in required:
        for lon, lat in ingredient_points(item):
            lons.append(_unwrap_lon(float(lon), origin))
            lats.append(float(lat))

    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    lon_span = max(0.2, east - west)
    lat_span = max(0.2, north - south)
    pad_lon = max(0.08, lon_span * padding_fraction)
    pad_lat = max(0.05, lat_span * padding_fraction)
    west -= pad_lon
    east += pad_lon
    south -= pad_lat
    north += pad_lat

    pts = [
        [_wrap_lon(east), south],
        [_wrap_lon(west), south],
        [_wrap_lon(west), north],
        [_wrap_lon(east), north],
    ]
    return (
        RegionBPoly(pts, bpoly.offshore_azimuth_deg, edge_labels=bpoly.edge_labels),
        "Expanded four-sided bpoly to encompass all required feature boxes without using the initial candidate as a required envelope anchor.",
    )


def features_to_geojson(features_doc: dict[str, Any]) -> dict[str, Any]:
    geo_features = []
    for item in features_doc.get("features", []):
        if item.get("type") != "bbox":
            continue
        west, south, east, north = item["geometry"]
        coords = [[west, south], [east, south], [east, north], [west, north], [west, south]]
        geo_features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": item.get("id"),
                    "label": item.get("label"),
                    "role": item.get("role"),
                    "category": item.get("category"),
                    "required": item.get("required", True),
                    "notes": item.get("notes", ""),
                    "guard_distance_km": item.get("guard_distance_km"),
                    "blocks_final_pass": item.get("blocks_final_pass"),
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    return {
        "type": "FeatureCollection",
        "name": "target_region_feature_polygons",
        "features": geo_features,
    }


def is_complex_feature_request(request: dict[str, Any] | str, features_doc: dict[str, Any]) -> tuple[bool, list[str]]:
    text = normalize_request_text(request)
    reasons: list[str] = []
    for term in [
        "archipelago",
        "island chain",
        "aleut",
        "all tidal",
        "all the tidal",
        "tidal energy",
        "connectivity",
        "connected",
        "wave climate",
    ]:
        if term in text:
            reasons.append(f"prompt contains complex-scope term '{term}'")
            break
    categories = {f.get("category") for f in features_doc.get("features", [])}
    if "geopolitical_separation" in categories:
        reasons.append("feature plan includes geopolitical separation context")
    if "island_archipelago_completeness" in categories:
        reasons.append("feature plan includes island/archipelago completeness")
    if "lake_connecting_channels" in categories:
        reasons.append("feature plan includes lake connecting-channel context")
    river_like = [f for f in features_doc.get("features", []) if f.get("category") in {"upstream_river_extent", "channel_connectivity"}]
    if len(river_like) >= 3:
        reasons.append("feature plan includes multiple river/channel connectivity boxes")
    return bool(reasons), reasons
