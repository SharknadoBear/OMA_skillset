#!/usr/bin/env python3
"""Offline regression tests for bounded CO-OPS residual station screening."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import geopandas as gpd
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from screen_tidal_stations import screen_stations


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        contract = root / "contract.json"
        contract.write_text(json.dumps({
            "schema_version": "fvcom_open_exterior_contract_v2",
            "residual_components": [{
                "segment_id": 0,
                "geometry_lonlat": [[-75.16, 38.23], [-75.15, 38.24]],
            }],
        }), encoding="utf-8")
        wet = root / "wet.gpkg"
        gpd.GeoDataFrame(
            [{"name": "wet", "geometry": Polygon([(-75.5, 38.0), (-74.8, 38.0), (-74.8, 38.6), (-75.5, 38.6)])}],
            crs="EPSG:4326",
        ).to_file(wet, layer="wet_domain", driver="GPKG")
        fixture = root / "stations.json"
        fixture.write_text(json.dumps({"stations": [
            {
                "id": "8570283", "name": "Ocean City Inlet", "lat": 38.32833, "lng": -75.09167,
                "tidal": True, "products": ["Water Levels", "Tide Predictions"],
                "datums_available": True, "harmonic_constituents_available": True,
            },
            {
                "id": "river", "name": "River gauge", "lat": 38.25, "lng": -75.14,
                "tidal": False, "products": ["Water Levels"],
                "datums_available": True, "harmonic_constituents_available": False,
            },
            {
                "id": "far", "name": "Far station", "lat": 39.5, "lng": -75.1,
                "tidal": True, "products": ["Water Levels"],
                "datums_available": True, "harmonic_constituents_available": True,
            },
        ]}), encoding="utf-8")
        result = screen_stations(contract, wet_domain_gpkg=wet, radius_km=25.0, fixture_json=fixture)
        assert result["policy"]["river_gauges_allowed"] is False
        assert result["eligible_station_count"] == 1
        candidates = result["components"][0]["candidates"]
        assert candidates[0]["station_id"] == "8570283"
        assert candidates[0]["eligible_for_residual_obc"] is True
        assert next(item for item in candidates if item["station_id"] == "river")["eligible_for_residual_obc"] is False
        assert all(item["station_id"] != "far" for item in candidates)
    print("passed bounded NOAA CO-OPS station-screen tests")


if __name__ == "__main__":
    main()
