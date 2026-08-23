#!/usr/bin/env python3
"""Screen nearby NOAA CO-OPS tidal stations for residual-boundary forcing eligibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import requests
from shapely.geometry import LineString, Point, Polygon, shape
from shapely.ops import unary_union


MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"
ELIGIBLE_PRODUCTS = {"water levels", "tide predictions"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, target)


def haversine_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = first
    lon2, lat2 = second
    radius_km = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    value = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return float(2.0 * radius_km * math.asin(math.sqrt(value)))


def _request_json(url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout_s, headers={"User-Agent": "OMA-noaa-coops-tides/1"})
    response.raise_for_status()
    return response.json()


def _component_midpoint(component: dict[str, Any]) -> tuple[float, float]:
    coords = component.get("geometry_lonlat") or []
    if len(coords) < 2:
        raise ValueError("Every residual component requires at least two geometry_lonlat coordinates")
    point = LineString(coords).interpolate(0.5, normalized=True)
    return float(point.x), float(point.y)


def _load_wet_geometry(path: str | Path | None):
    if not path:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Wet-domain source not found: {source}")
    import geopandas as gpd

    for layer in ("wet_domain", "model_domain_polygon", "resolved_domain_polygon"):
        try:
            values = gpd.read_file(source, layer=layer)
        except Exception:
            continue
        if values.empty:
            continue
        if values.crs is None:
            values = values.set_crs("EPSG:4326")
        values = values.to_crs("EPSG:4326")
        polygons = [geom for geom in values.geometry if geom is not None and not geom.is_empty]
        if polygons:
            return unary_union(polygons)
    raise ValueError("Wet-domain source has no wet/model/resolved polygon layer")


def _same_wet_component(wet_geometry, component_point: Point, station_point: Point) -> bool:
    if wet_geometry is None:
        return False
    polygons = (
        [wet_geometry]
        if isinstance(wet_geometry, Polygon)
        else [part for part in getattr(wet_geometry, "geoms", []) if isinstance(part, Polygon)]
    )
    # A one-kilometre-scale angular tolerance accommodates shoreline/source
    # registration while still requiring the same retained water component.
    tolerance_deg = 0.01
    return any(
        polygon.buffer(tolerance_deg).covers(component_point)
        and polygon.buffer(tolerance_deg).covers(station_point)
        for polygon in polygons
    )


def _live_inventory() -> list[dict[str, Any]]:
    payload = _request_json(f"{MDAPI}/stations.json?type=waterlevels")
    return list(payload.get("stations") or [])


def _live_station_details(station: dict[str, Any]) -> dict[str, Any]:
    station_id = str(station["id"])
    metadata = _request_json(f"{MDAPI}/stations/{station_id}.json").get("stations", [{}])[0]
    products = _request_json(f"{MDAPI}/stations/{station_id}/products.json").get("products", [])
    datums = _request_json(f"{MDAPI}/stations/{station_id}/datums.json").get("datums", [])
    constituents = _request_json(f"{MDAPI}/stations/{station_id}/harcon.json").get("HarmonicConstituents", [])
    return {
        **station,
        **metadata,
        "products": [str(item.get("name", "")) for item in products],
        "datums_available": bool(datums),
        "harmonic_constituents_available": bool(constituents),
    }


def _fixture_inventory(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return list(payload.get("stations") or [])


def screen_stations(
    contract_path: str | Path,
    *,
    wet_domain_gpkg: str | Path | None,
    radius_km: float = 25.0,
    fixture_json: str | Path | None = None,
) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    components = list(contract.get("residual_components") or [])
    if radius_km <= 0.0 or radius_km > 100.0:
        raise ValueError("--radius-km must be in (0, 100]")
    wet_geometry = _load_wet_geometry(wet_domain_gpkg)
    inventory = _fixture_inventory(fixture_json) if fixture_json else _live_inventory()
    component_results: list[dict[str, Any]] = []
    detail_cache: dict[str, dict[str, Any]] = {}
    for component in components:
        midpoint = _component_midpoint(component)
        candidates: list[dict[str, Any]] = []
        for station in inventory:
            try:
                coordinate = (float(station["lng"]), float(station["lat"]))
            except (KeyError, TypeError, ValueError):
                continue
            distance_km = haversine_km(midpoint, coordinate)
            if distance_km > radius_km:
                continue
            station_id = str(station.get("id", ""))
            if not station_id:
                continue
            if station_id not in detail_cache:
                detail_cache[station_id] = dict(station) if fixture_json else _live_station_details(station)
            detail = detail_cache[station_id]
            products = {str(value).strip().lower() for value in detail.get("products", [])}
            tidal = bool(detail.get("tidal", False))
            product_ok = bool(products & ELIGIBLE_PRODUCTS)
            datum_ok = bool(detail.get("datums_available", False))
            harmonic_ok = bool(detail.get("harmonic_constituents_available", False))
            hydraulic = _same_wet_component(
                wet_geometry,
                Point(midpoint),
                Point(coordinate),
            )
            eligible = bool(tidal and product_ok and datum_ok and ("water levels" in products or harmonic_ok) and hydraulic)
            candidates.append({
                "station_id": station_id,
                "name": str(detail.get("name", "")),
                "longitude": coordinate[0],
                "latitude": coordinate[1],
                "distance_km": distance_km,
                "tidal": tidal,
                "products": sorted(products),
                "datums_available": datum_ok,
                "harmonic_constituents_available": harmonic_ok,
                "same_retained_wet_component": hydraulic,
                "eligible_for_residual_obc": eligible,
                "eligibility_failures": [
                    name
                    for name, passed in (
                        ("station_not_tidal", tidal),
                        ("water_level_or_prediction_product_missing", product_ok),
                        ("datum_metadata_missing", datum_ok),
                        ("water_level_or_harmonics_missing", "water levels" in products or harmonic_ok),
                        ("not_same_retained_wet_component", hydraulic),
                    )
                    if not passed
                ],
            })
        candidates.sort(key=lambda item: (not item["eligible_for_residual_obc"], item["distance_km"], item["station_id"]))
        component_results.append({
            "segment_id": int(component.get("segment_id", len(component_results))),
            "midpoint_lonlat": list(midpoint),
            "geometry_lonlat": component.get("geometry_lonlat"),
            "candidate_count": len(candidates),
            "eligible_candidate_count": sum(bool(item["eligible_for_residual_obc"]) for item in candidates),
            "candidates": candidates,
        })
    return {
        "schema_version": "noaa_coops_tidal_station_screen_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "provider": "NOAA CO-OPS",
            "metadata_api": MDAPI,
            "inventory_mode": "offline_fixture" if fixture_json else "live_metadata_api",
        },
        "policy": {
            "radius_km": float(radius_km),
            "station_type": "NOAA_COOPS_tidal_water_level_only",
            "river_gauges_allowed": False,
            "station_is_eligibility_not_automatic_obc": True,
            "hydraulic_connectivity_rule": "same_retained_wet_component",
        },
        "inputs": {
            "open_exterior_contract": str(contract_path),
            "wet_domain_gpkg": str(Path(wet_domain_gpkg).resolve()) if wet_domain_gpkg else None,
            "fixture_json": str(Path(fixture_json).resolve()) if fixture_json else None,
        },
        "source_contract_sha256": sha256_file(contract_path),
        "components": component_results,
        "eligible_station_count": sum(
            int(item["eligible_candidate_count"]) for item in component_results
        ),
    }


def _write_geojson(path: Path, result: dict[str, Any]) -> None:
    features = []
    for component in result["components"]:
        if component.get("geometry_lonlat"):
            features.append({
                "type": "Feature",
                "properties": {"kind": "residual_water_segment", "segment_id": component["segment_id"]},
                "geometry": {"type": "LineString", "coordinates": component["geometry_lonlat"]},
            })
        features.append({
            "type": "Feature",
            "properties": {"kind": "residual_midpoint", "segment_id": component["segment_id"]},
            "geometry": {"type": "Point", "coordinates": component["midpoint_lonlat"]},
        })
        for station in component["candidates"]:
            features.append({
                "type": "Feature",
                "properties": {
                    "kind": "coops_station",
                    "segment_id": component["segment_id"],
                    **{key: value for key, value in station.items() if key not in {"longitude", "latitude", "products", "eligibility_failures"}},
                },
                "geometry": {"type": "Point", "coordinates": [station["longitude"], station["latitude"]]},
            })
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _plot(path: Path, result: dict[str, Any], wet_domain_gpkg: str | Path | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    wet = _load_wet_geometry(wet_domain_gpkg)
    if wet is not None:
        geoms = [wet] if isinstance(wet, Polygon) else list(getattr(wet, "geoms", []))
        for geom in geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, color="#8ecae6", alpha=0.35, zorder=0)
            ax.plot(x, y, color="#31572c", linewidth=0.8, zorder=1)
    for component in result["components"]:
        if component.get("geometry_lonlat"):
            line = LineString(component["geometry_lonlat"])
            x, y = line.xy
            ax.plot(x, y, color="#ff8c00", linewidth=4.0, zorder=3, label="residual water segment")
        lon, lat = component["midpoint_lonlat"]
        ax.scatter([lon], [lat], marker="x", s=90, color="#ff7f00", zorder=4, label="residual midpoint")
        for station in component["candidates"]:
            color = "#007f5f" if station["eligible_for_residual_obc"] else "#777777"
            ax.scatter([station["longitude"]], [station["latitude"]], marker="^", s=70, color=color, zorder=4)
            ax.text(station["longitude"], station["latitude"], f" {station['station_id']} {station['distance_km']:.1f} km", fontsize=8)
    ax.set_title("NOAA CO-OPS residual-boundary station screen")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-exterior-contract", required=True)
    parser.add_argument("--wet-domain-gpkg", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius-km", type=float, default=25.0)
    parser.add_argument("--offline-fixture")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = screen_stations(
        args.open_exterior_contract,
        wet_domain_gpkg=args.wet_domain_gpkg,
        radius_km=args.radius_km,
        fixture_json=args.offline_fixture,
    )
    geojson = output_dir / "coops_station_screen.geojson"
    map_path = output_dir / "coops_station_screen_map.png"
    result_path = output_dir / "noaa_coops_tidal_station_screen_v1.json"
    _write_geojson(geojson, result)
    _plot(map_path, result, args.wet_domain_gpkg)
    result["outputs"] = {
        "station_geojson": str(geojson),
        "station_geojson_sha256": sha256_file(geojson),
        "station_map": str(map_path),
        "station_map_sha256": sha256_file(map_path),
        "station_screen_json": str(result_path),
    }
    atomic_json(result_path, result)
    print(json.dumps({"eligible_station_count": result["eligible_station_count"], "output": str(result_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
