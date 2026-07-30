#!/usr/bin/env python3
"""Automatically rebuild the Long Island Sound two-gate boundary package.

The July 2 ``e`` package is read only as a negative rejection fixture.  Its
full-resolution GSHHS level-1 land polygons are the sole shoreline source.
This helper searches deterministic shoreline samples for the shortest valid
water cross-section in each mission corridor, constructs the wet domain from
source shoreline paths and the two gates, and emits an adaptive-coastal-v2
compatible boundary-resolution package plus validation evidence.

Bathymetry is intentionally outside this preparation helper.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Sequence

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, substring, transform, unary_union


PROJECTED_EPSG = 32618
PROFILE = "adaptive-coastal-v2"
SAMPLE_SPACING_M = 25.0
BOUNDARY_MAX_SPACING_M = 250.0
LAND_TARGET_M = 250.0
OPEN_ANCHOR_TARGET_M = 250.0
OPEN_CENTRAL_TARGET_M = 8_000.0
OPEN_GRADATION = 0.15
MIN_EXTERIOR_OVERLAP = 0.98
ENDPOINT_SNAP_TOLERANCE_M = 0.05
LAND_CROSSING_TOLERANCE_M = 0.05
MIN_COMPONENT_AREA_M2 = 100.0

NEGATIVE_RELATIVE = Path(
    "Workspace/Preprocessing/fvcom-bdry-arc/runs/"
    "subagent_series_bdry_arc_16case_20260702/cases/"
    "long_island_sound_bdry_arc/e"
)
DEFAULT_OUTPUT_RELATIVE = Path(
    "Workspace/Preprocessing/fvcom-grid-generation/runs/"
    "gmsh_six_case_20260729T2150PDT/04_long_island_sound/preparation"
)

# Windows are scientific route constraints, not interactive selections.
GATE_WINDOWS_WGS84 = {
    "west_east_river": (-73.88, 40.75, -73.70, 40.90),
    "east_race": (-72.55, 41.05, -72.15, 41.40),
}
REQUIRED_FEATURES_WGS84 = {
    "western_sound_exchange_context": (-73.65, 40.90),
    "long_island_sound_core": (-72.90, 41.05),
    "eastern_sound_exchange_context": (-72.35, 41.15),
}
MAINLAND_SEED_WGS84 = (-72.70, 41.50)
LONG_ISLAND_SEED_WGS84 = (-73.10, 40.85)


@dataclass(frozen=True)
class CandidateMetric:
    rank: int
    length_m: float
    endpoint_a_snap_m: float
    endpoint_b_snap_m: float
    land_crossing_m: float
    valid: bool
    rejection: str | None


@dataclass(frozen=True)
class GateSelection:
    chain_id: str
    line_xy: LineString
    window_wgs84: tuple[float, float, float, float]
    candidate_pool_count: int
    evaluated_candidate_count: int
    valid_candidate_count: int
    selected_rank: int
    selected: CandidateMetric
    evidence_lines: tuple[LineString, ...]
    evidence_metrics: tuple[CandidateMetric, ...]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _workspace_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "Agent_skill_dev").is_dir() and (
            candidate / "Workspace"
        ).is_dir():
            return candidate.resolve()
    raise RuntimeError("Cannot locate workspace root containing Agent_skill_dev/Workspace")


def _sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _projector() -> tuple[Transformer, Transformer]:
    return (
        Transformer.from_crs("EPSG:4326", f"EPSG:{PROJECTED_EPSG}", always_xy=True),
        Transformer.from_crs(f"EPSG:{PROJECTED_EPSG}", "EPSG:4326", always_xy=True),
    )


def _project_geometry(geometry: Any, transformer: Transformer) -> Any:
    return transform(transformer.transform, geometry)


def _polygon_components(geometry: Any) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return [
            part
            for item in geometry.geoms
            for part in _polygon_components(item)
        ]
    return []


def _line_components(geometry: Any) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return [
            part
            for item in geometry.geoms
            for part in _line_components(item)
        ]
    return []


def _land_polygon_for_seed(
    land: gpd.GeoDataFrame,
    seed_wgs84: tuple[float, float],
    to_xy: Transformer,
) -> tuple[str, Polygon]:
    seed = _project_geometry(Point(seed_wgs84), to_xy)
    containing = land[land.geometry.covers(seed)]
    if containing.empty:
        distances = land.geometry.distance(seed)
        row = land.loc[distances.idxmin()]
    else:
        row = containing.iloc[0]
    geometry = row.geometry
    if not isinstance(geometry, Polygon):
        raise RuntimeError(f"Selected source land ID {row['id']} is not a Polygon")
    return str(row["id"]), geometry


def _sample_linework(
    linework: Any,
    spacing_m: float,
) -> tuple[np.ndarray, list[Point]]:
    points: list[Point] = []
    for line in _line_components(linework):
        if line.length <= 0.0:
            continue
        distances = np.arange(0.0, float(line.length), float(spacing_m))
        distances = np.append(distances, float(line.length))
        points.extend(line.interpolate(float(distance)) for distance in distances)
    if not points:
        raise RuntimeError("Gate search window contains no shoreline samples")
    coordinates = np.asarray([[point.x, point.y] for point in points], dtype=float)
    return coordinates, points


def _trimmed_line(line: LineString, trim_m: float = 2.0) -> LineString:
    if line.length <= 2.0 * trim_m:
        return line
    result = substring(line, trim_m, line.length - trim_m)
    if not isinstance(result, LineString):
        return line
    return result


def _candidate_metric(
    line: LineString,
    bank_a_boundary: Any,
    bank_b_boundary: Any,
    land_union: Any,
    rank: int,
) -> CandidateMetric:
    start = Point(line.coords[0])
    end = Point(line.coords[-1])
    snap_a = float(start.distance(bank_a_boundary))
    snap_b = float(end.distance(bank_b_boundary))
    crossing = float(_trimmed_line(line).intersection(land_union).length)
    rejection: str | None = None
    if snap_a > ENDPOINT_SNAP_TOLERANCE_M:
        rejection = "bank_a_endpoint_not_snapped"
    elif snap_b > ENDPOINT_SNAP_TOLERANCE_M:
        rejection = "bank_b_endpoint_not_snapped"
    elif crossing > LAND_CROSSING_TOLERANCE_M:
        rejection = "positive_length_land_crossing"
    return CandidateMetric(
        rank=rank,
        length_m=float(line.length),
        endpoint_a_snap_m=snap_a,
        endpoint_b_snap_m=snap_b,
        land_crossing_m=crossing,
        valid=rejection is None,
        rejection=rejection,
    )


def _search_gate(
    chain_id: str,
    window_wgs84: tuple[float, float, float, float],
    bank_a: Polygon,
    bank_b: Polygon,
    land_union: Any,
    to_xy: Transformer,
) -> GateSelection:
    search_window = _project_geometry(box(*window_wgs84), to_xy)
    shore_a = bank_a.exterior.intersection(search_window)
    shore_b = bank_b.exterior.intersection(search_window)
    if shore_a.is_empty or shore_b.is_empty:
        raise RuntimeError(f"{chain_id}: source banks do not enter search window")

    coords_a, points_a = _sample_linework(shore_a, SAMPLE_SPACING_M)
    coords_b, points_b = _sample_linework(shore_b, SAMPLE_SPACING_M)
    tree = cKDTree(coords_b)
    neighbor_count = min(64, len(coords_b))
    distances, indices = tree.query(coords_a, k=neighbor_count)
    if neighbor_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    candidate_pairs: dict[tuple[int, int], tuple[float, Point, Point]] = {}
    exact_a, exact_b = nearest_points(shore_a, shore_b)
    exact_key = (
        int(round(exact_a.x * 1000.0)),
        int(round(exact_b.x * 1000.0)),
    )
    candidate_pairs[exact_key] = (
        float(exact_a.distance(exact_b)),
        exact_a,
        exact_b,
    )
    for index_a in range(len(points_a)):
        for distance, index_b in zip(
            np.asarray(distances[index_a]).reshape(-1),
            np.asarray(indices[index_a]).reshape(-1),
        ):
            point_a = points_a[index_a]
            point_b = points_b[int(index_b)]
            key = (
                int(round(point_a.x * 1000.0)) * 10_000_000
                + int(round(point_a.y * 1000.0)),
                int(round(point_b.x * 1000.0)) * 10_000_000
                + int(round(point_b.y * 1000.0)),
            )
            candidate_pairs.setdefault(
                key,
                (float(distance), point_a, point_b),
            )

    ordered_pairs = sorted(candidate_pairs.values(), key=lambda item: item[0])
    evaluated: list[tuple[CandidateMetric, LineString]] = []
    selected: tuple[CandidateMetric, LineString] | None = None
    # Evaluating the shortest 5,000 sampled pairs is deterministic and spans
    # substantially more than the local 25 m discretization uncertainty.
    for rank, (_distance, point_a, point_b) in enumerate(
        ordered_pairs[:5_000],
        start=1,
    ):
        line = LineString([point_a, point_b])
        metric = _candidate_metric(
            line,
            bank_a.exterior,
            bank_b.exterior,
            land_union,
            rank,
        )
        evaluated.append((metric, line))
        if metric.valid and (
            selected is None or metric.length_m < selected[0].length_m
        ):
            selected = (metric, line)
    if selected is None:
        taxonomy = sorted(
            {item.rejection for item, _line in evaluated if item.rejection}
        )
        raise RuntimeError(
            f"{chain_id}: no valid gate in deterministic candidate set; "
            f"rejections={taxonomy}"
        )

    valid_count = sum(metric.valid for metric, _line in evaluated)
    evidence = evaluated[: min(40, len(evaluated))]
    selected_metric, selected_line = selected
    return GateSelection(
        chain_id=chain_id,
        line_xy=selected_line,
        window_wgs84=window_wgs84,
        candidate_pool_count=len(ordered_pairs),
        evaluated_candidate_count=len(evaluated),
        valid_candidate_count=valid_count,
        selected_rank=selected_metric.rank,
        selected=selected_metric,
        evidence_lines=tuple(line for _metric, line in evidence),
        evidence_metrics=tuple(metric for metric, _line in evidence),
    )


def _concatenate_lines(parts: Sequence[LineString]) -> LineString:
    coordinates: list[tuple[float, float]] = []
    for part in parts:
        values = list(part.coords)
        if not values:
            continue
        if coordinates:
            if Point(coordinates[-1]).distance(Point(values[0])) > 0.10:
                raise RuntimeError("Shoreline path parts do not share an endpoint")
            values = values[1:]
        coordinates.extend((float(x), float(y)) for x, y in values)
    return LineString(coordinates)


def _forward_ring_path(
    ring: LineString,
    start_distance: float,
    end_distance: float,
) -> LineString:
    length = float(ring.length)
    if end_distance >= start_distance:
        result = substring(ring, start_distance, end_distance)
        if not isinstance(result, LineString):
            raise RuntimeError("Degenerate shoreline substring")
        return result
    return _concatenate_lines(
        [
            substring(ring, start_distance, length),
            substring(ring, 0.0, end_distance),
        ]
    )


def _two_ring_paths(
    polygon: Polygon,
    start: Point,
    end: Point,
) -> tuple[LineString, LineString]:
    ring = LineString(polygon.exterior.coords)
    start_distance = float(ring.project(start))
    end_distance = float(ring.project(end))
    forward = _forward_ring_path(ring, start_distance, end_distance)
    reverse_forward = _forward_ring_path(ring, end_distance, start_distance)
    backward = LineString(list(reverse_forward.coords)[::-1])
    return forward, backward


def _ring_from_paths(
    mainland_path: LineString,
    east_gate: LineString,
    long_island_path: LineString,
    west_gate: LineString,
) -> Polygon:
    ring = _concatenate_lines(
        [
            mainland_path,
            east_gate,
            long_island_path,
            LineString(list(west_gate.coords)[::-1]),
        ]
    )
    coordinates = list(ring.coords)
    if Point(coordinates[0]).distance(Point(coordinates[-1])) > 0.10:
        raise RuntimeError("Constructed LIS ring does not close")
    return Polygon(coordinates)


def _build_wet_domain(
    mainland: Polygon,
    long_island: Polygon,
    west_gate: LineString,
    east_gate: LineString,
    land_union: Any,
    required_features_xy: dict[str, Point],
) -> tuple[Polygon, dict[str, Any]]:
    west_mainland = Point(west_gate.coords[0])
    west_long_island = Point(west_gate.coords[-1])
    east_mainland = Point(east_gate.coords[0])
    east_long_island = Point(east_gate.coords[-1])
    mainland_paths = _two_ring_paths(mainland, west_mainland, east_mainland)
    long_island_paths = _two_ring_paths(
        long_island,
        east_long_island,
        west_long_island,
    )

    evaluated: list[dict[str, Any]] = []
    accepted: list[tuple[float, float, Polygon, dict[str, Any]]] = []
    for north_index, mainland_path in enumerate(mainland_paths):
        for south_index, long_island_path in enumerate(long_island_paths):
            record: dict[str, Any] = {
                "mainland_path_index": north_index,
                "long_island_path_index": south_index,
                "mainland_path_length_m": float(mainland_path.length),
                "long_island_path_length_m": float(long_island_path.length),
            }
            try:
                raw = _ring_from_paths(
                    mainland_path,
                    east_gate,
                    long_island_path,
                    west_gate,
                )
            except RuntimeError as exc:
                record.update({"accepted": False, "reason": str(exc)})
                evaluated.append(record)
                continue
            record["raw_valid"] = bool(raw.is_valid)
            record["raw_area_m2"] = float(raw.area)
            if not raw.is_valid or raw.area <= 0.0:
                record.update({"accepted": False, "reason": "invalid_raw_ring"})
                evaluated.append(record)
                continue
            water = raw.difference(land_union)
            components = [
                item
                for item in _polygon_components(water)
                if item.area >= MIN_COMPONENT_AREA_M2
            ]
            record["wet_component_count"] = len(components)
            if len(components) != 1:
                record.update(
                    {"accepted": False, "reason": "wet_component_count_not_one"}
                )
                evaluated.append(record)
                continue
            wet = components[0]
            containment = {
                name: bool(wet.covers(point))
                for name, point in required_features_xy.items()
            }
            record["required_feature_containment"] = containment
            if not all(containment.values()):
                record.update(
                    {"accepted": False, "reason": "required_feature_outside"}
                )
                evaluated.append(record)
                continue
            overlaps = {
                "west_east_river": float(
                    west_gate.intersection(wet.boundary).length / west_gate.length
                ),
                "east_race": float(
                    east_gate.intersection(wet.boundary).length / east_gate.length
                ),
            }
            record["gate_exterior_overlap_fraction"] = overlaps
            if min(overlaps.values()) < MIN_EXTERIOR_OVERLAP:
                record.update(
                    {"accepted": False, "reason": "gate_exterior_overlap_below_0.98"}
                )
                evaluated.append(record)
                continue
            wet = orient(wet, sign=1.0)
            record.update(
                {
                    "accepted": True,
                    "reason": None,
                    "wet_area_m2": float(wet.area),
                    "hole_count": len(wet.interiors),
                }
            )
            evaluated.append(record)
            accepted.append((float(raw.area), float(wet.area), wet, record))
    if not accepted:
        raise RuntimeError(
            "No shoreline-path combination produced one valid feature-complete "
            f"wet component: {evaluated}"
        )
    accepted.sort(key=lambda item: (item[0], item[1]))
    _raw_area, _wet_area, selected, selected_record = accepted[0]
    return selected, {
        "selection_policy": (
            "smallest valid source-shoreline enclosure yielding one water "
            "component containing all required LIS mission points"
        ),
        "evaluated_path_combinations": evaluated,
        "selected_path_combination": selected_record,
    }


def _densify_ring(
    coordinates: Iterable[Sequence[float]],
    max_spacing_m: float,
) -> np.ndarray:
    values = np.asarray([[float(item[0]), float(item[1])] for item in coordinates])
    if np.linalg.norm(values[0] - values[-1]) <= 1.0e-8:
        values = values[:-1]
    output: list[np.ndarray] = []
    for index, start in enumerate(values):
        end = values[(index + 1) % len(values)]
        length = float(np.linalg.norm(end - start))
        pieces = max(1, int(math.ceil(length / max_spacing_m)))
        for piece in range(pieces):
            output.append(start + (piece / pieces) * (end - start))
    return np.asarray(output, dtype=float)


def _contiguous_open_runs(kinds: Sequence[str]) -> list[tuple[int, ...]]:
    flags = [item == "open" for item in kinds]
    if not any(flags):
        return []
    starts = [
        index
        for index, flag in enumerate(flags)
        if flag and not flags[(index - 1) % len(flags)]
    ]
    runs: list[tuple[int, ...]] = []
    for start in starts:
        run: list[int] = []
        index = start
        while flags[index]:
            run.append(index)
            index = (index + 1) % len(flags)
        runs.append(tuple(run))
    return runs


def _adaptive_boundary_rows(
    wet_domain: Polygon,
    gates: dict[str, LineString],
) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]], np.ndarray, tuple[str, ...]]:
    gate_union = unary_union(list(gates.values()))
    exterior = _densify_ring(
        wet_domain.exterior.coords,
        BOUNDARY_MAX_SPACING_M,
    )
    vertex_kinds = tuple(
        "open"
        if Point(coordinate).distance(gate_union) <= 0.05
        else "land"
        for coordinate in exterior
    )
    segment_kinds = tuple(
        "open"
        if vertex_kinds[index] == "open"
        and vertex_kinds[(index + 1) % len(vertex_kinds)] == "open"
        else "land"
        for index in range(len(vertex_kinds))
    )
    runs = _contiguous_open_runs(segment_kinds)
    if len(runs) != 2:
        raise RuntimeError(
            f"Resolved exterior produced {len(runs)} open runs, expected exactly two"
        )

    rows: list[dict[str, Any]] = []
    chain_records: list[dict[str, Any]] = []
    global_index = 0
    cumulative = 0.0
    open_anchor_counter = 0
    for position, coordinate in enumerate(exterior):
        point = Point(coordinate)
        kind = vertex_kinds[position]
        previous = exterior[(position - 1) % len(exterior)]
        if position > 0:
            cumulative += float(np.linalg.norm(coordinate - previous))
        hard = kind == "open" and (
            vertex_kinds[(position - 1) % len(vertex_kinds)] != "open"
            or vertex_kinds[(position + 1) % len(vertex_kinds)] != "open"
        )
        source_chain = "land"
        target = LAND_TARGET_M
        if kind == "open":
            distances = {
                chain_id: float(point.distance(line))
                for chain_id, line in gates.items()
            }
            source_chain = min(distances, key=distances.get)
            gate = gates[source_chain]
            position_on_gate = float(gate.project(point))
            distance_to_landfall = min(
                position_on_gate,
                float(gate.length) - position_on_gate,
            )
            target = min(
                OPEN_CENTRAL_TARGET_M,
                OPEN_ANCHOR_TARGET_M
                + OPEN_GRADATION * max(distance_to_landfall, 0.0),
            )
        anchor_id = ""
        anchor_type = ""
        if hard:
            anchor_id = f"open_landfall_{open_anchor_counter:04d}"
            anchor_type = "open_landfall"
            open_anchor_counter += 1
        rows.append(
            {
                "node_index_zero_based": global_index,
                "chain_id": 0,
                "chain_position": position,
                "boundary_kind": kind,
                "target_spacing_m": float(target),
                "is_hard_anchor": bool(hard),
                "anchor_type": anchor_type,
                "anchor_id": anchor_id,
                "source_chain": source_chain,
                "source_position_m": float(cumulative),
                "geometry": point,
            }
        )
        global_index += 1
    chain_records.append(
        {
            "chain_id": 0,
            "kind": "outer",
            "node_count": len(exterior),
            "start_node_index_zero_based": 0,
            "end_node_index_zero_based": len(exterior) - 1,
            "hard_anchor_count": open_anchor_counter,
            "open_landfall_hard_anchor_count": open_anchor_counter,
        }
    )

    for chain_id, interior in enumerate(wet_domain.interiors, start=1):
        island = _densify_ring(interior.coords, BOUNDARY_MAX_SPACING_M)
        start_index = global_index
        cumulative = 0.0
        for position, coordinate in enumerate(island):
            if position > 0:
                cumulative += float(np.linalg.norm(coordinate - island[position - 1]))
            rows.append(
                {
                    "node_index_zero_based": global_index,
                    "chain_id": chain_id,
                    "chain_position": position,
                    "boundary_kind": "island",
                    "target_spacing_m": LAND_TARGET_M,
                    "is_hard_anchor": False,
                    "anchor_type": "",
                    "anchor_id": "",
                    "source_chain": f"island_{chain_id:04d}",
                    "source_position_m": float(cumulative),
                    "geometry": Point(coordinate),
                }
            )
            global_index += 1
        chain_records.append(
            {
                "chain_id": chain_id,
                "kind": "island",
                "node_count": len(island),
                "start_node_index_zero_based": start_index,
                "end_node_index_zero_based": global_index - 1,
            }
        )

    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{PROJECTED_EPSG}")
    return frame, chain_records, exterior, vertex_kinds


def _edge_target_metrics(nodes: gpd.GeoDataFrame) -> dict[str, float]:
    ratios: list[float] = []
    gradations: list[float] = []
    for _chain_id, group in nodes.groupby("chain_id", sort=True):
        group = group.sort_values("chain_position")
        coordinates = np.asarray([[item.x, item.y] for item in group.geometry])
        targets = group["target_spacing_m"].to_numpy(dtype=float)
        lengths = np.linalg.norm(np.roll(coordinates, -1, axis=0) - coordinates, axis=1)
        ratios.extend((lengths / np.maximum(targets, 1.0e-12)).tolist())
        gradations.extend(
            (
                np.abs(np.roll(targets, -1) - targets)
                / np.maximum(lengths, 1.0e-12)
            ).tolist()
        )
    return {
        "maximum_edge_to_target_ratio": round(float(np.max(ratios)), 12),
        "p95_edge_to_target_ratio": round(float(np.quantile(ratios, 0.95)), 12),
        "maximum_target_gradation": round(float(np.max(gradations)), 12),
    }


def _write_adaptive_package(
    output_dir: Path,
    wet_domain_xy: Polygon,
    gates_xy: dict[str, LineString],
    to_lonlat: Transformer,
    validation: dict[str, Any],
    input_paths: dict[str, Path],
) -> dict[str, Path]:
    gpkg = output_dir / "boundary_resolution.gpkg"
    nodes_geojson = output_dir / "boundary_resolution_nodes.geojson"
    diagnostics_path = output_dir / "boundary_resolution_diagnostics.json"
    manifest_path = output_dir / "boundary_resolution_manifest.json"
    nodes, chains, exterior, vertex_kinds = _adaptive_boundary_rows(
        wet_domain_xy,
        gates_xy,
    )
    nodes_wgs84 = nodes.to_crs("EPSG:4326")
    domain_wgs84 = _project_geometry(wet_domain_xy, to_lonlat)
    gates_wgs84 = {
        key: _project_geometry(value, to_lonlat)
        for key, value in gates_xy.items()
    }
    islands_xy = [Polygon(interior) for interior in wet_domain_xy.interiors]
    islands_wgs84 = [
        _project_geometry(value, to_lonlat)
        for value in islands_xy
    ]

    gpd.GeoDataFrame(
        [{"profile": PROFILE, "geometry": domain_wgs84}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(gpkg, layer="resolved_domain_polygon", driver="GPKG")
    gpd.GeoDataFrame(
        [
            {
                "chain_id": chain_id,
                "segment_class": "open_boundary",
                "kind": "ocean_exchange",
                "cyclic": False,
                "orientation": "source",
                "geometry": geometry,
            }
            for chain_id, geometry in gates_wgs84.items()
        ],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(gpkg, layer="resolved_open_boundary", driver="GPKG")
    if islands_wgs84:
        gpd.GeoDataFrame(
            [
                {
                    "resolved_island_id": index,
                    "shape_class": "gshhs_full_resolution_retained",
                    "protected_mission": False,
                    "source_area_m2": float(islands_xy[index].area),
                    "generalized_area_m2": float(islands_xy[index].area),
                    "generalized_area_error_fraction": 0.0,
                    "target_spacing_m": LAND_TARGET_M,
                    "final_target_spacing_m": LAND_TARGET_M,
                    "resolved_vertex_count": len(interior.coords) - 1,
                    "resolved_area_m2": float(islands_xy[index].area),
                    "geometry": geometry,
                }
                for index, (interior, geometry) in enumerate(
                    zip(wet_domain_xy.interiors, islands_wgs84)
                )
            ],
            geometry="geometry",
            crs="EPSG:4326",
        ).to_file(gpkg, layer="resolved_island_polygons", driver="GPKG")
    nodes_wgs84.to_file(gpkg, layer="boundary_nodes", driver="GPKG")
    nodes_wgs84.to_file(nodes_geojson, driver="GeoJSON")

    edge_metrics = _edge_target_metrics(nodes)
    open_nodes = int((nodes["boundary_kind"] == "open").sum())
    island_nodes = int((nodes["boundary_kind"] == "island").sum())
    diagnostics = {
        "schema_version": "lis_two_gate_boundary_resolution_diagnostics_v1",
        "created_utc": _utc_now(),
        "profile": PROFILE,
        "boundary_node_count": len(nodes),
        "open_boundary_node_count": open_nodes,
        "island_boundary_node_count": island_nodes,
        "outer_boundary_node_count": len(exterior),
        "resolved_island_count": len(islands_xy),
        "outer_vertex_kind_counts": {
            kind: vertex_kinds.count(kind) for kind in sorted(set(vertex_kinds))
        },
        "chains": chains,
        "edge_target_metrics": edge_metrics,
        "validation": validation,
    }
    _write_json(diagnostics_path, diagnostics)

    qa = {
        "open_boundary_node_count": open_nodes,
        "island_boundary_node_count": island_nodes,
        "total_boundary_node_count": len(nodes),
        "resolved_island_count": len(islands_xy),
        "source_island_count": len(islands_xy),
        "topology_absolute_area_change_fraction": 0.0,
        "protected_mission_operation_count": 0,
        "open_arc_land_intersection_m": validation["gate_validation"][
            "total_land_crossing_m"
        ],
        "open_arc_exterior_overlap_fraction": validation["gate_validation"][
            "minimum_exterior_overlap_fraction"
        ],
        "resolved_domain_valid": bool(wet_domain_xy.is_valid),
        **edge_metrics,
        "hard_anchor_count": int(nodes["is_hard_anchor"].sum()),
        "open_landfall_hard_anchor_count": int(nodes["is_hard_anchor"].sum()),
        "wet_component_count": 1,
        "required_feature_containment_fraction": validation[
            "required_features"
        ]["containment_fraction"],
    }
    manifest = {
        "schema_version": "fvcom_boundary_resolution_manifest_v2",
        "name": "long_island_sound_two_gate_gmsh_research",
        "created_utc": _utc_now(),
        "created_by": Path(__file__).name,
        "profile": PROFILE,
        "final_status": "pass",
        "failure_taxonomy": [],
        "advisory_taxonomy": [
            "research_only_gmsh_boundary_preparation",
            "bathymetry_not_prepared",
        ],
        "inputs": {
            key: str(path.resolve()) for key, path in input_paths.items()
        },
        "settings": {
            "profile": PROFILE,
            "land_spacing_m": LAND_TARGET_M,
            "mission_spacing_m": LAND_TARGET_M,
            "open_anchor_spacing_m": OPEN_ANCHOR_TARGET_M,
            "open_central_spacing_m": OPEN_CENTRAL_TARGET_M,
            "gradation": OPEN_GRADATION,
            "source_boundary_max_spacing_m": BOUNDARY_MAX_SPACING_M,
            "gate_candidate_sample_spacing_m": SAMPLE_SPACING_M,
            "gate_policy": "automatic_shortest_valid_shoreline_to_shoreline",
        },
        "qa": qa,
        "open_boundaries": [
            {
                "chain_id": chain_id,
                "kind": "ocean_exchange",
                "cyclic": False,
                "orientation": "source",
                "length_m": float(gates_xy[chain_id].length),
            }
            for chain_id in ("east_race", "west_east_river")
        ],
        "chains": chains,
        "outputs": {
            "boundary_resolution_gpkg": str(gpkg.resolve()),
            "boundary_resolution_diagnostics_json": str(diagnostics_path.resolve()),
            "boundary_resolution_nodes_geojson": str(nodes_geojson.resolve()),
            "boundary_resolution_review_map": str(
                (output_dir / "lis_two_gate_overview_map.png").resolve()
            ),
            "boundary_resolution_manifest": str(manifest_path.resolve()),
        },
    }
    _write_json(manifest_path, manifest)
    return {
        "gpkg": gpkg,
        "nodes_geojson": nodes_geojson,
        "diagnostics": diagnostics_path,
        "manifest": manifest_path,
    }


def _negative_fixture_report(
    negative_dir: Path,
    land_union_xy: Any,
) -> dict[str, Any]:
    manifest_path = negative_dir / "bdry_arc_manifest.json"
    package_path = negative_dir / "bdry_arc_package.gpkg"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    arc = gpd.read_file(package_path, layer="open_boundary_arc").to_crs(
        f"EPSG:{PROJECTED_EPSG}"
    ).geometry.iloc[0]
    wet = gpd.read_file(package_path, layer="wet_domain").to_crs(
        f"EPSG:{PROJECTED_EPSG}"
    ).geometry.iloc[0]
    crossing = float(arc.intersection(land_union_xy).length)
    overlap = float(arc.intersection(wet.boundary).length / arc.length)
    return {
        "fixture_role": "negative_rejection_only",
        "immutable_source_directory": str(negative_dir.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_package_sha256": _sha256(package_path),
        "source_final_status": manifest.get("final_status"),
        "source_failure_taxonomy": manifest.get("failure_taxonomy", []),
        "recomputed": {
            "open_arc_length_m": float(arc.length),
            "open_arc_land_crossing_m": crossing,
            "open_arc_exterior_overlap_fraction": overlap,
            "wet_component_count": int(
                manifest.get("wet_domain", {}).get("face_count", 0)
            ),
        },
        "rejection_checks": {
            "land_crossing_zero": crossing <= LAND_CROSSING_TOLERANCE_M,
            "exterior_overlap_at_least_0_98": overlap >= MIN_EXTERIOR_OVERLAP,
            "one_wet_component": (
                int(manifest.get("wet_domain", {}).get("face_count", 0)) == 1
            ),
        },
        "accepted_for_reuse": False,
    }


def _plot_overview(
    path: Path,
    land_wgs84: gpd.GeoDataFrame,
    domain_wgs84: Polygon,
    gates_wgs84: dict[str, LineString],
) -> None:
    bounds = domain_wgs84.bounds
    frame = box(
        bounds[0] - 0.08,
        bounds[1] - 0.08,
        bounds[2] + 0.08,
        bounds[3] + 0.08,
    )
    clipped = land_wgs84.clip(frame)
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    clipped.plot(ax=axis, color="#d8d0bd", edgecolor="#4b4b4b", linewidth=0.35)
    gpd.GeoSeries([domain_wgs84], crs="EPSG:4326").plot(
        ax=axis,
        facecolor="#8ecae6",
        edgecolor="#023047",
        alpha=0.55,
        linewidth=1.0,
    )
    colors = {"west_east_river": "#d62828", "east_race": "#6a00f4"}
    for chain_id, gate in gates_wgs84.items():
        x, y = gate.xy
        axis.plot(x, y, color=colors[chain_id], linewidth=3.0, label=chain_id)
        axis.scatter(
            [x[0], x[-1]],
            [y[0], y[-1]],
            color=colors[chain_id],
            edgecolor="white",
            s=35,
            zorder=5,
        )
    for name, (lon, lat) in REQUIRED_FEATURES_WGS84.items():
        axis.scatter(lon, lat, marker="*", s=80, color="#ffb703", edgecolor="black")
        axis.annotate(name, (lon, lat), xytext=(4, 4), textcoords="offset points")
    axis.set_title("Long Island Sound automatic two-gate wet domain")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend(loc="lower center", ncol=2)
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_gate_search(
    path: Path,
    land_wgs84: gpd.GeoDataFrame,
    selections: dict[str, GateSelection],
    to_lonlat: Transformer,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    colors = {"west_east_river": "#d62828", "east_race": "#6a00f4"}
    for axis, chain_id in zip(
        axes,
        ("west_east_river", "east_race"),
    ):
        selection = selections[chain_id]
        window = box(*selection.window_wgs84)
        land_wgs84.clip(window).plot(
            ax=axis,
            color="#ded6c4",
            edgecolor="#333333",
            linewidth=0.5,
        )
        for candidate in selection.evidence_lines:
            lonlat = _project_geometry(candidate, to_lonlat)
            x, y = lonlat.xy
            axis.plot(x, y, color="#f4a261", alpha=0.25, linewidth=0.7)
        selected = _project_geometry(selection.line_xy, to_lonlat)
        x, y = selected.xy
        axis.plot(x, y, color=colors[chain_id], linewidth=3.0)
        axis.scatter(
            [x[0], x[-1]],
            [y[0], y[-1]],
            color=colors[chain_id],
            edgecolor="white",
            s=35,
            zorder=5,
        )
        west, south, east, north = selection.window_wgs84
        axis.set_xlim(west, east)
        axis.set_ylim(south, north)
        axis.set_title(
            f"{chain_id}\nselected {selection.selected.length_m / 1000.0:.2f} km; "
            f"rank {selection.selected_rank}"
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _plot_negative_fixture(
    path: Path,
    negative_dir: Path,
) -> None:
    package = negative_dir / "bdry_arc_package.gpkg"
    wet = gpd.read_file(package, layer="wet_domain")
    arc = gpd.read_file(package, layer="open_boundary_arc")
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    wet.plot(
        ax=axis,
        facecolor="#bde0fe",
        edgecolor="#023047",
        alpha=0.55,
    )
    arc.plot(ax=axis, color="#d00000", linewidth=2.0, label="failed e open arc")
    axis.set_title("Immutable failed e package: negative rejection fixture")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend()
    axis.set_aspect("equal", adjustable="box")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def prepare(workspace_root: Path, output_dir: Path) -> dict[str, Any]:
    negative_dir = (workspace_root / NEGATIVE_RELATIVE).resolve()
    coastline_gpkg = negative_dir / "coastline" / "lis_gshhs_land.gpkg"
    coastline_manifest = negative_dir / "coastline" / "lis_gshhs_manifest.json"
    failed_manifest = negative_dir / "bdry_arc_manifest.json"
    failed_package = negative_dir / "bdry_arc_package.gpkg"
    for path in (
        coastline_gpkg,
        coastline_manifest,
        failed_manifest,
        failed_package,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing non-immutable preparation rerun into nonempty {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    to_xy, to_lonlat = _projector()
    land_wgs84 = gpd.read_file(coastline_gpkg, layer="land_polygons").to_crs(
        "EPSG:4326"
    )
    land_xy = land_wgs84.to_crs(f"EPSG:{PROJECTED_EPSG}")
    land_union_xy = land_xy.geometry.union_all()
    mainland_id, mainland = _land_polygon_for_seed(
        land_xy,
        MAINLAND_SEED_WGS84,
        to_xy,
    )
    long_island_id, long_island = _land_polygon_for_seed(
        land_xy,
        LONG_ISLAND_SEED_WGS84,
        to_xy,
    )
    if mainland_id == long_island_id:
        raise RuntimeError("Automatic bank selection chose one polygon for both banks")

    selections = {
        chain_id: _search_gate(
            chain_id,
            window,
            mainland,
            long_island,
            land_union_xy,
            to_xy,
        )
        for chain_id, window in GATE_WINDOWS_WGS84.items()
    }
    gates_xy = {
        chain_id: selection.line_xy
        for chain_id, selection in selections.items()
    }
    required_features_xy = {
        name: _project_geometry(Point(lonlat), to_xy)
        for name, lonlat in REQUIRED_FEATURES_WGS84.items()
    }
    wet_domain_xy, path_selection = _build_wet_domain(
        mainland,
        long_island,
        gates_xy["west_east_river"],
        gates_xy["east_race"],
        land_union_xy,
        required_features_xy,
    )
    wet_domain_wgs84 = _project_geometry(wet_domain_xy, to_lonlat)

    gate_metrics: dict[str, dict[str, Any]] = {}
    for chain_id, selection in selections.items():
        gate = selection.line_xy
        endpoint_snaps = [
            float(Point(gate.coords[0]).distance(mainland.exterior)),
            float(Point(gate.coords[-1]).distance(long_island.exterior)),
        ]
        crossing = float(_trimmed_line(gate).intersection(land_union_xy).length)
        overlap = float(gate.intersection(wet_domain_xy.boundary).length / gate.length)
        gate_metrics[chain_id] = {
            "length_m": float(gate.length),
            "endpoint_lonlat": [
                list(
                    to_lonlat.transform(
                        float(gate.coords[0][0]),
                        float(gate.coords[0][1]),
                    )
                ),
                list(
                    to_lonlat.transform(
                        float(gate.coords[-1][0]),
                        float(gate.coords[-1][1]),
                    )
                ),
            ],
            "maximum_endpoint_snap_m": max(endpoint_snaps),
            "land_crossing_m": crossing,
            "exterior_overlap_fraction": overlap,
            "candidate_search": {
                "sample_spacing_m": SAMPLE_SPACING_M,
                "candidate_pool_count": selection.candidate_pool_count,
                "evaluated_candidate_count": selection.evaluated_candidate_count,
                "valid_candidate_count": selection.valid_candidate_count,
                "selected_rank": selection.selected_rank,
                "selection_policy": (
                    "shortest valid straight shoreline-to-shoreline candidate "
                    "in the deterministic scientific corridor"
                ),
            },
        }

    containment = {
        name: bool(wet_domain_xy.covers(point))
        for name, point in required_features_xy.items()
    }
    validation = {
        "schema_version": "lis_two_gate_validation_v1",
        "created_utc": _utc_now(),
        "status": "pass",
        "projected_crs": f"EPSG:{PROJECTED_EPSG}",
        "source_bank_selection": {
            "mainland_gshhs_id": mainland_id,
            "long_island_gshhs_id": long_island_id,
            "selection": "source polygons containing fixed scientific land seeds",
        },
        "gate_validation": {
            "expected_gate_count": 2,
            "actual_gate_count": len(gates_xy),
            "gates": gate_metrics,
            "maximum_endpoint_snap_m": max(
                item["maximum_endpoint_snap_m"] for item in gate_metrics.values()
            ),
            "endpoint_snap_tolerance_m": ENDPOINT_SNAP_TOLERANCE_M,
            "total_land_crossing_m": sum(
                item["land_crossing_m"] for item in gate_metrics.values()
            ),
            "land_crossing_tolerance_m": LAND_CROSSING_TOLERANCE_M,
            "minimum_exterior_overlap_fraction": min(
                item["exterior_overlap_fraction"] for item in gate_metrics.values()
            ),
            "required_minimum_exterior_overlap_fraction": MIN_EXTERIOR_OVERLAP,
        },
        "wet_domain": {
            "valid": bool(wet_domain_xy.is_valid),
            "wet_component_count": 1,
            "area_m2": float(wet_domain_xy.area),
            "perimeter_m": float(wet_domain_xy.length),
            "island_hole_count": len(wet_domain_xy.interiors),
            "bounds_wgs84": list(wet_domain_wgs84.bounds),
        },
        "required_features": {
            "definitions_wgs84": REQUIRED_FEATURES_WGS84,
            "contained": containment,
            "contained_count": sum(containment.values()),
            "required_count": len(containment),
            "containment_fraction": (
                sum(containment.values()) / max(len(containment), 1)
            ),
        },
        "shoreline_path_selection": path_selection,
        "hard_gates": {
            "endpoint_snap": max(
                item["maximum_endpoint_snap_m"] for item in gate_metrics.values()
            )
            <= ENDPOINT_SNAP_TOLERANCE_M,
            "zero_land_crossing": sum(
                item["land_crossing_m"] for item in gate_metrics.values()
            )
            <= LAND_CROSSING_TOLERANCE_M,
            "exterior_overlap": min(
                item["exterior_overlap_fraction"] for item in gate_metrics.values()
            )
            >= MIN_EXTERIOR_OVERLAP,
            "one_wet_component": True,
            "required_feature_containment": all(containment.values()),
        },
    }
    if not all(validation["hard_gates"].values()):
        validation["status"] = "needs_review"
        raise RuntimeError(f"LIS two-gate validation failed: {validation['hard_gates']}")

    validation_path = _write_json(
        output_dir / "lis_two_gate_validation.json",
        validation,
    )
    negative_report = _negative_fixture_report(negative_dir, land_union_xy)
    negative_path = _write_json(
        output_dir / "lis_failed_e_negative_fixture_report.json",
        negative_report,
    )

    candidate_rows: list[dict[str, Any]] = []
    for chain_id, selection in selections.items():
        for metric, line in zip(
            selection.evidence_metrics,
            selection.evidence_lines,
        ):
            candidate_rows.append(
                {
                    "chain_id": chain_id,
                    "candidate_rank": metric.rank,
                    "length_m": metric.length_m,
                    "valid": metric.valid,
                    "rejection": metric.rejection or "",
                    "selected": metric.rank == selection.selected_rank,
                    "geometry": _project_geometry(line, to_lonlat),
                }
            )
        candidate_rows.append(
            {
                "chain_id": chain_id,
                "candidate_rank": selection.selected_rank,
                "length_m": selection.selected.length_m,
                "valid": True,
                "rejection": "",
                "selected": True,
                "geometry": _project_geometry(selection.line_xy, to_lonlat),
            }
        )
    candidates_path = output_dir / "lis_two_gate_candidates.geojson"
    gpd.GeoDataFrame(
        candidate_rows,
        geometry="geometry",
        crs="EPSG:4326",
    ).drop_duplicates(
        subset=["chain_id", "candidate_rank", "selected"],
    ).to_file(candidates_path, driver="GeoJSON")

    overview_path = output_dir / "lis_two_gate_overview_map.png"
    gate_search_path = output_dir / "lis_two_gate_gate_search_map.png"
    negative_map_path = output_dir / "lis_failed_e_negative_fixture_map.png"
    gates_wgs84 = {
        key: _project_geometry(value, to_lonlat)
        for key, value in gates_xy.items()
    }
    _plot_overview(
        overview_path,
        land_wgs84,
        wet_domain_wgs84,
        gates_wgs84,
    )
    _plot_gate_search(
        gate_search_path,
        land_wgs84,
        selections,
        to_lonlat,
    )
    _plot_negative_fixture(negative_map_path, negative_dir)

    input_paths = {
        "negative_fixture_manifest": failed_manifest,
        "negative_fixture_package": failed_package,
        "gshhs_coastline_gpkg": coastline_gpkg,
        "gshhs_coastline_manifest": coastline_manifest,
    }
    package_paths = _write_adaptive_package(
        output_dir,
        wet_domain_xy,
        gates_xy,
        to_lonlat,
        validation,
        input_paths,
    )
    preparation_manifest = {
        "schema_version": "lis_two_gate_preparation_manifest_v1",
        "created_utc": _utc_now(),
        "created_by": Path(__file__).name,
        "status": "pass",
        "scope": "boundary preparation only; bathymetry not fetched",
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "inputs": {
            key: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for key, path in input_paths.items()
        },
        "validation_summary": {
            "gate_count": 2,
            "maximum_endpoint_snap_m": validation["gate_validation"][
                "maximum_endpoint_snap_m"
            ],
            "total_land_crossing_m": validation["gate_validation"][
                "total_land_crossing_m"
            ],
            "minimum_exterior_overlap_fraction": validation["gate_validation"][
                "minimum_exterior_overlap_fraction"
            ],
            "wet_component_count": 1,
            "required_feature_containment_fraction": validation[
                "required_features"
            ]["containment_fraction"],
            "negative_fixture_accepted": False,
        },
        "outputs": {
            "boundary_resolution_manifest": str(
                package_paths["manifest"].resolve()
            ),
            "boundary_resolution_gpkg": str(package_paths["gpkg"].resolve()),
            "boundary_resolution_diagnostics": str(
                package_paths["diagnostics"].resolve()
            ),
            "boundary_resolution_nodes_geojson": str(
                package_paths["nodes_geojson"].resolve()
            ),
            "validation_json": str(validation_path.resolve()),
            "candidate_geojson": str(candidates_path.resolve()),
            "overview_map": str(overview_path.resolve()),
            "gate_search_map": str(gate_search_path.resolve()),
            "negative_fixture_report": str(negative_path.resolve()),
            "negative_fixture_map": str(negative_map_path.resolve()),
        },
    }
    preparation_path = _write_json(
        output_dir / "lis_two_gate_preparation_manifest.json",
        preparation_manifest,
    )
    preparation_manifest["outputs"]["preparation_manifest"] = str(
        preparation_path.resolve()
    )
    _write_json(preparation_path, preparation_manifest)
    return preparation_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the automatic Long Island Sound East River/Race two-gate "
            "adaptive-v2 boundary preparation package."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=_workspace_root())
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace_root = args.workspace_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (workspace_root / DEFAULT_OUTPUT_RELATIVE).resolve()
    )
    try:
        manifest = prepare(workspace_root, output_dir)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "output_dir": str(output_dir),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
