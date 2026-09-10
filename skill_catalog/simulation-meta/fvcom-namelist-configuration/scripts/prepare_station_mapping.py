#!/usr/bin/env python3
"""Map eligible NOAA CO-OPS stations to exact FVCOM cells and local depths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mesh(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, int, int, int]]]:
    nodes: dict[int, tuple[float, float, float]] = {}
    elements: list[tuple[int, int, int, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            parts = raw.split()
            if not parts:
                continue
            if parts[0].upper() == "ND":
                if len(parts) < 5:
                    raise ValueError(f"Malformed ND at {path}:{line_number}")
                nodes[int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))
            elif parts[0].upper() == "E3T":
                if len(parts) < 5:
                    raise ValueError(f"Malformed E3T at {path}:{line_number}")
                elements.append(tuple(map(int, parts[1:5])))
    if not nodes or not elements:
        raise ValueError("Mesh must contain nodes and triangular elements")
    return nodes, elements


def barycentric(
    point: tuple[float, float], triangle: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float] | None:
    x, y = point
    a, b, c = triangle
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if denominator == 0.0:
        return None
    u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    return u, v, 1.0 - u - v


def locate(
    point: tuple[float, float],
    nodes: dict[int, tuple[float, float, float]],
    elements: list[tuple[int, int, int, int]],
) -> tuple[int, float]:
    x, y = point
    matches: list[tuple[int, float]] = []
    for element_id, n1, n2, n3 in elements:
        triangle = (nodes[n1], nodes[n2], nodes[n3])
        if x < min(item[0] for item in triangle) or x > max(item[0] for item in triangle):
            continue
        if y < min(item[1] for item in triangle) or y > max(item[1] for item in triangle):
            continue
        weights = barycentric(point, triangle)
        if weights is None or min(weights) < -1.0e-10:
            continue
        depth = sum(weight * abs(vertex[2]) for weight, vertex in zip(weights, triangle, strict=True))
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError(f"Station maps to nonpositive depth in element {element_id}")
        matches.append((element_id, depth))
    if not matches:
        raise ValueError("Station marked eligible is not inside any FVCOM element")
    return min(matches, key=lambda item: item[0])


def station_coordinates(longitude: float, latitude: float, mesh_crs: str) -> tuple[float, float]:
    key = mesh_crs.strip().upper().replace(" ", "")
    if key in {"EPSG:4326", "EPSG:4269"}:
        return longitude, latitude
    try:
        from pyproj import Transformer  # type: ignore
    except ImportError as exc:
        raise ValueError(f"Mapping stations into {mesh_crs!r} requires pyproj") from exc
    return tuple(map(float, Transformer.from_crs("EPSG:4326", mesh_crs, always_xy=True).transform(longitude, latitude)))


WATER_POLICIES = ("strict_inside", "containing_cell_or_nearest_wet_cell")


def centroid_geometry(nodes, elements, mesh_crs: str, metric_crs: str | None):
    from pyproj import CRS, Transformer
    native = CRS.from_user_input(mesh_crs)
    to_geo = Transformer.from_crs(native, "EPSG:4326", always_xy=True)
    ids = list(nodes)
    lon, lat = to_geo.transform([nodes[n][0] for n in ids], [nodes[n][1] for n in ids])
    if metric_crs is None:
        # Regional UTM is the deterministic default; callers may supply another metric CRS.
        longitude, latitude = sum(lon) / len(lon), sum(lat) / len(lat)
        zone = max(1, min(60, int((longitude + 180) // 6) + 1))
        metric_crs = f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"
    metric = CRS.from_user_input(metric_crs)
    if not metric.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-12 for a in metric.axis_info[:2]):
        raise ValueError("Proxy distance CRS must be projected with metre axes")
    project = Transformer.from_crs(native, metric, always_xy=True)
    east, north = project.transform([nodes[n][0] for n in ids], [nodes[n][1] for n in ids])
    xy = dict(zip(ids, zip(east, north)))
    geographic = Transformer.from_crs(metric, "EPSG:4326", always_xy=True)
    point_project = Transformer.from_crs("EPSG:4326", metric, always_xy=True)
    cells = {}
    for cell, a, b, c in elements:
        x, y = sum(xy[n][0] for n in (a,b,c))/3, sum(xy[n][1] for n in (a,b,c))/3
        depth = sum(abs(nodes[n][2]) for n in (a,b,c))/3
        if not all(math.isfinite(v) for v in (x,y,depth)) or depth <= 0:
            continue
        cells[cell] = (x, y, depth, geographic.transform(x,y), [a,b,c])
    if not cells:
        raise ValueError("No finite positive-depth model cells for station sampling")
    return cells, point_project, metric.to_string()


def build(grid_contract: str | Path, station_inventory: str | Path, output: str | Path,
          water_level_mapping_policy: str | None = None, metric_crs: str | None = None,
          proxy_review: str | Path | None = None,
          diagnostic_mapping: str | Path | None = None) -> dict[str, Any]:
    contract_path = Path(grid_contract).expanduser().resolve()
    inventory_path = Path(station_inventory).expanduser().resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    policy = water_level_mapping_policy or inventory.get("water_level_mapping_policy", "strict_inside")
    if policy not in WATER_POLICIES:
        raise ValueError("Unsupported water-level mapping policy")
    if contract.get("schema_version") != "fvcom_grid_delivery_contract_v1":
        raise ValueError("Unsupported grid delivery contract")
    if inventory.get("schema") != "noaa_coops_validation_station_inventory_v1":
        raise ValueError("Unsupported station inventory")
    mesh_spec = contract.get("mesh")
    if not isinstance(mesh_spec, dict):
        raise ValueError("Grid contract has no mesh object")
    mesh_path = Path(str(mesh_spec["path"]))
    if not mesh_path.is_absolute():
        mesh_path = contract_path.parent / mesh_path
    mesh_path = mesh_path.resolve()
    mesh_hash = sha256_file(mesh_path)
    if mesh_hash != str(mesh_spec["sha256"]).lower():
        raise ValueError("Grid contract mesh hash mismatch")
    if str(inventory.get("mesh_sha256", "")).lower() != mesh_hash:
        raise ValueError("Station inventory was not screened against this exact mesh")
    nodes, elements = read_mesh(mesh_path)
    if len(nodes) != int(mesh_spec["node_count"]) or len(elements) != int(mesh_spec["triangle_count"]):
        raise ValueError("Grid contract mesh counts do not match the file")

    mapped: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    cells = project = distance_crs = None
    if policy != "strict_inside":
        cells, project, distance_crs = centroid_geometry(nodes, elements, str(mesh_spec["coordinate_crs"]), metric_crs)
    review = json.loads(Path(proxy_review).read_text(encoding="utf-8-sig")) if proxy_review else None
    if review and (review.get("schema") != "fvcom_waterlevel_proxy_review_v1" or review.get("status") != "passed" or review.get("mesh_sha256") != mesh_hash):
        raise ValueError("Proxy review must pass and bind this exact mesh")
    reviewed = {str(r["station_id"]): r for r in review.get("stations", [])} if review else {}
    if review and len(reviewed) != len(review.get('stations', [])):
        raise ValueError('Proxy review has duplicate station IDs')
    pending_reviews = []
    for station in inventory.get("stations", []):
        if not isinstance(station, dict):
            continue
        station_id = str(station.get("id", "")).strip()
        role = str(station.get("role", ""))
        outside_only = station.get("exclusion_reason") == "outside_actual_wet_mesh"
        allow_proxy = role == "water_level" and policy != "strict_inside"
        if not station.get("eligible") and not (allow_proxy and outside_only):
            excluded.append({"station_id": station_id, "reason": str(station.get("exclusion_reason") or "not_eligible")})
            continue
        longitude = float(station["longitude"])
        latitude = float(station["latitude"])
        point = station_coordinates(longitude, latitude, str(mesh_spec["coordinate_crs"]))
        try:
            cell_id, depth = locate(point, nodes, elements)
            inside = True
        except ValueError as exc:
            if not allow_proxy or str(exc) != "Station marked eligible is not inside any FVCOM element":
                raise
            inside = False
            x, y = project.transform(longitude, latitude)
            cell_id = min(cells, key=lambda k: ((cells[k][0]-x)**2+(cells[k][1]-y)**2, k))
            depth = cells[cell_id][2]
        row = {
                "station_id": station_id,
                "longitude": longitude,
                "latitude": latitude,
                "cell_id": cell_id,
                "depth_m": depth,
                "role": role,
                "name": station.get("name"),
            }
        if allow_proxy:
            x, y = project.transform(longitude, latitude)
            cx, cy, depth, (clon, clat), node_ids = cells[cell_id]
            row.update(longitude=clon, latitude=clat, depth_m=depth, validation_eligible=True,
                       observation_longitude=longitude, observation_latitude=latitude,
                       inside_wet_mesh=inside,
                       spatial_mapping={"method": "containing_cell_centroid" if inside else "nearest_wet_cell_centroid",
                           "observation_longitude": longitude, "observation_latitude": latitude,
                           "model_longitude": clon, "model_latitude": clat,
                           "distance_m": math.hypot(cx-x, cy-y), "metric_crs": distance_crs,
                           "cell_id": cell_id, "node_ids": node_ids, "node_weights": [1/3]*3,
                           "depth_m": depth, "inside_wet_mesh": inside,
                           "sampling": "arithmetic_mean_of_three_nodal_elevations",
                           "selection_basis": "geometry_only_before_agreement_statistics"})
            if not inside:
                if review:
                    from pyproj import CRS
                    record = reviewed.get(station_id, {})
                    required = {'cell_id', 'observation_longitude', 'observation_latitude',
                                'model_longitude', 'model_latitude', 'node_ids', 'depth_m',
                                'distance_m', 'selection_approved'}
                    if not required <= record.keys() or not review.get('metric_crs'):
                        raise ValueError('Proxy review lacks coordinate/geometry approval binding: ' + station_id)
                    rx, ry = project.transform(record['model_longitude'], record['model_latitude'])
                    if (record['selection_approved'] is not True or record['cell_id'] != cell_id
                            or record['observation_longitude'] != longitude or record['observation_latitude'] != latitude
                            or CRS.from_user_input(review['metric_crs']) != CRS.from_user_input(distance_crs)
                            or sorted(record['node_ids']) != sorted(node_ids)
                            or not math.isfinite(float(record['depth_m'])) or abs(record['depth_m'] - depth) > 1e-4
                            or not math.isfinite(float(record['distance_m'])) or abs(record['distance_m'] - math.hypot(cx-x, cy-y)) > .01
                            or not all(math.isfinite(v) for v in (rx, ry)) or math.hypot(rx-cx, ry-cy) > .01):
                        raise ValueError("Proxy review and deterministic selected cell/coordinates disagree: " + station_id)
                if not review:
                    pending_reviews.append(station_id)
                else:
                    row["spatial_mapping"]["geometry_review_sha256"] = sha256_file(proxy_review)
        mapped.append(row)
    if not mapped:
        raise ValueError("No eligible stations mapped into the wet mesh")
    mapped.sort(key=lambda item: (str(item["role"]), str(item["station_id"])))
    if diagnostic_mapping:
        diagnostic = json.loads(Path(diagnostic_mapping).read_text(encoding="utf-8-sig"))
        if diagnostic.get("mesh_sha256") != mesh_hash:
            raise ValueError("Diagnostic mapping must bind the exact mesh")
        rows = diagnostic.get("stations", [])
        if any(r.get("role") != "model_diagnostic" or r.get("validation_eligible") is not False for r in rows):
            raise ValueError("Retained diagnostic rows must be explicitly non-comparison diagnostics")
        mapped = rows + mapped
    if len({r["station_id"] for r in mapped}) != len(mapped):
        raise ValueError("Duplicate station ID in combined mapping")
    payload = {
        "schema": "fvcom_station_mapping_v1",
        "status": "needs_proxy_review" if pending_reviews else "ready",
        "grid_contract_sha256": sha256_file(contract_path),
        "mesh_sha256": mesh_hash,
        "station_inventory_sha256": sha256_file(inventory_path),
        "selection_policy": "geometry-based cell stations; water policy explicit; currents require strict wet containment and downward profiles",
        "water_level_mapping_policy": policy,
        "proxy_review_pending": pending_reviews,
        "diagnostic_mapping_sha256": sha256_file(diagnostic_mapping) if diagnostic_mapping else None,
        "station_count": len(mapped),
        "stations": mapped,
        "excluded": excluded,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-contract", required=True)
    parser.add_argument("--station-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--water-level-mapping-policy", choices=WATER_POLICIES)
    parser.add_argument("--metric-crs")
    parser.add_argument("--proxy-review")
    parser.add_argument("--diagnostic-mapping")
    args = parser.parse_args()
    print(json.dumps(build(args.grid_contract, args.station_inventory, args.output,
        args.water_level_mapping_policy, args.metric_crs, args.proxy_review, args.diagnostic_mapping), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
