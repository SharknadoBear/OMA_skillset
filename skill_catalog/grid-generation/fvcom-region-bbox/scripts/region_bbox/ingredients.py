from __future__ import annotations

import re
from typing import Any


def request_text(request: dict[str, Any] | str) -> str:
    if isinstance(request, str):
        return request
    chunks = []
    for key in ("region", "mission", "purpose", "description", "request", "text", "prompt"):
        val = request.get(key)
        if val:
            chunks.append(str(val))
    if not chunks:
        chunks.append(" ".join(str(v) for v in request.values() if isinstance(v, (str, int, float))))
    return " ".join(chunks)


def required_ingredients(request: dict[str, Any] | str) -> list[dict[str, Any]]:
    text = request_text(request).lower()
    items: list[dict[str, Any]] = []

    def bbox(i, label, role, geom, required=True, notes=""):
        items.append({"id": i, "label": label, "role": role, "type": "bbox", "geometry": geom, "required": required, "notes": notes})

    if "puget" in text or "salish" in text:
        broad = any(k in text for k in ["tidal energy", "all tidal", "all the tidal", "connect", "salish"])
        bbox("puget_sound", "Puget Sound", "target_water_body", [-123.35, 46.95, -122.15, 48.55])
        bbox("hood_canal", "Hood Canal", "tidal_channel", [-123.35, 47.25, -122.7, 48.15])
        bbox("strait_juan_de_fuca", "Strait of Juan de Fuca", "offshore_forcing_corridor", [-125.1, 48.0, -122.8, 48.6])
        if broad:
            bbox("san_juan_islands", "San Juan Islands", "mission_critical_connected_tidal_channels", [-123.35, 48.35, -122.65, 48.9])
            bbox("haro_boundary_pass", "Haro Strait and Boundary Pass", "mission_critical_connected_tidal_channels", [-123.35, 48.45, -122.75, 49.15])
            bbox("strait_georgia", "Strait of Georgia", "mission_critical_connected_tidal_channels", [-124.0, 48.8, -122.5, 50.35])
        return items

    if "long island" in text or "hypoxia" in text:
        bbox("lis_core", "Long Island Sound core", "target_water_body", [-73.95, 40.75, -71.83, 41.37])
        bbox("ny_harbor", "New York Harbor and East River", "connected_waterbody", [-74.35, 40.35, -73.72, 40.85])
        bbox("raritan_newark", "Raritan/Newark/Jamaica Bay context", "connected_waterbody", [-74.35, 40.35, -73.7, 40.78])
        bbox("race_block_island", "The Race and Block Island Sound", "eastern_open_ocean_context", [-72.35, 40.95, -71.45, 41.35])
        bbox("narragansett", "Narragansett Bay/Providence context", "connected_waterbody", [-71.55, 41.25, -71.05, 41.85])
        return items

    if "murderkill" in text:
        bbox("murderkill", "Murderkill River and mouth", "target_estuary", [-75.52, 39.0, -75.35, 39.18])
        bbox("delaware_bay_connection", "Delaware Bay connection", "open_bay_context", [-75.5, 38.9, -75.15, 39.25])
        return items

    if "delaware" in text:
        bbox("delaware_estuary", "Delaware River estuary", "target_water_body", [-75.8, 38.8, -74.8, 40.2])
        bbox("upstream_nontidal", "Upstream/nontidal river context", "river_input_context", [-75.3, 39.7, -74.6, 40.35])
        bbox("offshore_shelf", "Offshore tidal forcing apron", "offshore_buffer", [-75.5, 37.6, -73.5, 39.0])
        return items

    if "chesapeake" in text:
        bbox("chesapeake_bay", "Chesapeake Bay", "target_water_body", [-77.6, 36.7, -75.3, 39.7])
        bbox("offshore_midatlantic", "Atlantic forcing apron", "offshore_buffer", [-76.2, 36.2, -74.2, 38.5])
        return items

    if "aleut" in text or "aleuc" in text:
        bbox("aleutian_chain", "Aleutian Islands chain context", "required_geospatial_extent", [172.0, 51.0, -158.0, 55.5])
        return items

    if "hawaiian islands" in text or "hawaii islands" in text or "hawaii state" in text:
        bbox("hawaiian_chain", "Hawaiian Islands chain", "island_chain", [-161.0, 18.5, -154.5, 23.0])
        return items

    if "hawaii island" in text or "big island" in text or "otec" in text:
        bbox("hawaii_big_island", "Hawaii Island / Big Island", "target_island", [-156.2, 18.6, -154.6, 20.4])
        return items

    if "lake superior" in text:
        bbox("lake_superior", "Lake Superior", "lake", [-92.4, 46.2, -84.2, 49.2])
        return items
    if "lake ontario" in text:
        bbox("lake_ontario", "Lake Ontario", "lake", [-80.2, 43.0, -76.0, 44.6])
        return items
    if "lake erie" in text:
        bbox("lake_erie", "Lake Erie", "lake", [-83.6, 41.2, -78.5, 42.9])
        return items

    if "cook inlet" in text:
        bbox("cook_inlet", "Cook Inlet", "target_estuary", [-153.2, 58.7, -149.0, 61.5])
        bbox("gulf_of_alaska", "Gulf of Alaska forcing apron", "offshore_buffer", [-154.5, 57.0, -150.0, 59.5])
        return items

    if "southeast alaska" in text or "se ak" in text:
        bbox("se_alaska", "Southeast Alaska tidal channels", "target_region", [-139.8, 54.0, -128.0, 60.0])
        bbox("haida_gwaii_context", "Haida Gwaii context", "boundary_split_guard", [-133.5, 51.5, -130.0, 54.5])
        return items

    if "columbia" in text:
        bbox("columbia_estuary", "Columbia River estuary", "target_estuary", [-124.3, 45.8, -123.4, 46.45])
        bbox("offshore_columbia", "Pacific forcing apron", "offshore_buffer", [-125.0, 45.5, -123.8, 46.7])
        return items

    if "hudson" in text:
        bbox("hudson_estuary", "Hudson River estuary", "target_estuary", [-74.3, 40.5, -73.6, 42.2])
        bbox("ny_harbor", "New York Harbor / East River complexity", "needs_review_context", [-74.3, 40.45, -73.7, 40.9])
        return items

    if "san francisco" in text:
        bbox("sf_bay", "San Francisco Bay and Delta context", "target_estuary", [-123.0, 37.0, -121.3, 38.4])
        bbox("pacific_gate", "Pacific forcing apron", "offshore_buffer", [-123.3, 37.4, -122.3, 38.1])
        return items

    return items


def mission_scope_notes(request: dict[str, Any] | str) -> list[dict[str, Any]]:
    text = request_text(request).lower()
    notes: list[dict[str, Any]] = []
    if "puget" in text and any(k in text for k in ["tidal energy", "all tidal", "all the tidal", "connect"]):
        notes.append(
            {
                "gate": "mission_connectivity",
                "status": "requires_review",
                "message": "Puget tidal-energy/all-channel requests require Salish Sea connected-channel context unless explicitly scoped out.",
                "critical_context": ["San Juan Islands", "Strait of Georgia", "Boundary Pass", "Haro Strait", "Strait of Juan de Fuca"],
            }
        )
    if ("aleut" in text or "aleuc" in text) and "island" in text:
        notes.append(
            {
                "gate": "full_geospatial_extent",
                "status": "requires_review",
                "message": "Aleutian island-chain requests must include the intended chain extent or clearly name a scoped subregion, and maps must visibly show the RegionBox.",
            }
        )
    if re.search(r"\bhawaii island\b", text) and not any(k in text for k in ["hawaiian islands", "hawaii islands", "hawaii state"]):
        notes.append({"gate": "place_name_interpretation", "status": "assumed_big_island", "message": "Interpreting Hawaii Island as the Big Island."})
    if any(k in text for k in ["hawaiian islands", "hawaii islands", "hawaii state"]):
        notes.append({"gate": "place_name_interpretation", "status": "requires_island_chain", "message": "Interpreting request as the Hawaiian island chain/state."})
    return notes

