#!/usr/bin/env python3
"""Offline regression tests for bounded CO-OPS residual station screening."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import geopandas as gpd
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from screen_tidal_stations import screen_stations
from screen_tidal_stations import MDAPI


def test_live_catalog_discovery(root: Path, contract: Path) -> None:
    """Prediction-only sites must be discovered without weakening eligibility."""
    wet = root / "live_wet.gpkg"
    gpd.GeoDataFrame(
        [{"geometry": Polygon([(-75.2, 38.15), (-75.0, 38.15), (-75.0, 38.4), (-75.2, 38.4)])}],
        crs="EPSG:4326",
    ).to_file(wet, layer="wet_domain", driver="GPKG")
    ids = ["dual", "prediction", "no_datum", "no_harmonics", "river", "no_product", "disconnected", "far"]
    stations = {sid: {"id": sid, "name": sid, "lng": -75.14, "lat": 38.25} for sid in ids}
    stations["disconnected"]["lng"] = -75.30
    stations["far"]["lat"] = 39.5
    # A duplicate must retain the first catalog's coordinate and one candidate.
    duplicate = {**stations["dual"], "lng": -74.0}
    responses = {
        f"{MDAPI}/stations.json?type=waterlevels": {"stations": [stations["dual"], stations["river"]]},
        f"{MDAPI}/stations.json?type=tidepredictions": {
            "stations": [duplicate, *[stations[sid] for sid in ids if sid != "dual"]],
        },
    }
    for sid, station in stations.items():
        base = f"{MDAPI}/stations/{sid}"
        responses[base + ".json"] = {"stations": [{**station, "tidal": sid != "river"}]}
        products = ["Water Levels"] if sid in {"dual", "river"} else ["Tide Predictions"]
        responses[base + "/products.json"] = {"products": [{"name": p} for p in products] if sid != "no_product" else []}
        responses[base + "/datums.json"] = {"datums": [] if sid == "no_datum" else [{"name": "MSL"}]}
        responses[base + "/harcon.json"] = {"HarmonicConstituents": [] if sid in {"no_harmonics", "dual"} else [{"name": "M2"}]}
    with patch("screen_tidal_stations._request_json", side_effect=lambda url: responses[url]) as request:
        result = screen_stations(contract, wet_domain_gpkg=wet)
    candidates = result["components"][0]["candidates"]
    by_id = {item["station_id"]: item for item in candidates}
    assert "prediction" in by_id, "Live discovery omitted the prediction-only NOAA catalog"
    assert len(candidates) == len(by_id) == 7
    assert {sid for sid, item in by_id.items() if item["eligible_for_residual_obc"]} == {"dual", "prediction"}
    assert by_id["dual"]["longitude"] == -75.14
    assert by_id["dual"]["inventory_sources"] == [
        {"catalog_type": kind, "url": f"{MDAPI}/stations.json?type={kind}"}
        for kind in ("waterlevels", "tidepredictions")
    ]
    assert by_id["prediction"]["inventory_sources"][0]["catalog_type"] == "tidepredictions"
    for sid, failure in {
        "no_datum": "datum_metadata_missing", "no_harmonics": "water_level_or_harmonics_missing",
        "river": "station_not_tidal", "no_product": "water_level_or_prediction_product_missing",
        "disconnected": "not_same_retained_wet_component",
    }.items():
        assert failure in by_id[sid]["eligibility_failures"]
    urls = [call.args[0] for call in request.call_args_list]
    assert urls.count(f"{MDAPI}/stations/dual.json") == 1
    assert not any("/stations/far" in url for url in urls)
    assert result["source"]["inventory_station_count"] == 8
    # An unavailable catalog must remain an acquisition failure, not absence.
    def failing_request(url):
        if "type=tidepredictions" in url:
            raise RuntimeError("catalog unavailable")
        return responses[url]
    with patch("screen_tidal_stations._request_json", side_effect=failing_request):
        try:
            screen_stations(contract, wet_domain_gpkg=wet)
        except RuntimeError as error:
            assert str(error) == "catalog unavailable"
        else:
            raise AssertionError("Catalog acquisition failure was silently treated as no stations")


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
        test_live_catalog_discovery(root, contract)
    print("passed bounded NOAA CO-OPS station-screen tests")


if __name__ == "__main__":
    main()
