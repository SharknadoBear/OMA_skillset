"""Pure topology and schema helpers for the topobathy-flownet skill."""

from __future__ import annotations

import hashlib
import math
import struct
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

NODATA = -9999.0
SCHEMA_VERSION = "topobathy_flownet_v1"


Point = tuple[float, float]
Line = list[Point]
Sampler = Callable[[Sequence[Point]], Sequence[float]]


class WkbDecodeError(ValueError):
    """Raised for structurally unreadable WKB."""


def _wkb_header(data: bytes, offset: int) -> tuple[str, int, int, int]:
    if offset + 5 > len(data):
        raise WkbDecodeError("truncated WKB header")
    byte_order = data[offset]
    if byte_order == 0:
        endian = ">"
    elif byte_order == 1:
        endian = "<"
    else:
        raise WkbDecodeError(f"invalid WKB byte order {byte_order}")
    type_code = struct.unpack_from(f"{endian}I", data, offset + 1)[0]
    cursor = offset + 5
    has_z = bool(type_code & 0x80000000)
    has_m = bool(type_code & 0x40000000)
    has_srid = bool(type_code & 0x20000000)
    base_type = type_code & 0x000000FF if type_code & 0xE0000000 else type_code
    dimensions = 2 + int(has_z) + int(has_m)
    if not type_code & 0xE0000000:
        dimension_class, base_type = divmod(type_code, 1000)
        if dimension_class == 1:
            dimensions = 3
        elif dimension_class == 2:
            dimensions = 3
        elif dimension_class == 3:
            dimensions = 4
        elif dimension_class != 0:
            raise WkbDecodeError(f"unsupported WKB type code {type_code}")
    if has_srid:
        if cursor + 4 > len(data):
            raise WkbDecodeError("truncated EWKB SRID")
        cursor += 4
    return endian, int(base_type), dimensions, cursor


def _wkb_uint32(data: bytes, offset: int, endian: str) -> tuple[int, int]:
    if offset + 4 > len(data):
        raise WkbDecodeError("truncated WKB count")
    return struct.unpack_from(f"{endian}I", data, offset)[0], offset + 4


def _wkb_coordinates(
    data: bytes,
    offset: int,
    endian: str,
    dimensions: int,
    count: int,
) -> tuple[list[Point], bool, int]:
    coordinate_bytes = count * dimensions * 8
    if count > len(data) or offset + coordinate_bytes > len(data):
        raise WkbDecodeError("truncated or unreasonable WKB coordinate array")
    points: list[Point] = []
    all_finite = True
    format_string = f"{endian}{dimensions}d"
    for _ in range(count):
        values = struct.unpack_from(format_string, data, offset)
        offset += dimensions * 8
        all_finite = all_finite and all(math.isfinite(value) for value in values)
        points.append((float(values[0]), float(values[1])))
    return points, all_finite, offset


def decode_wkb_line_parts(raw_wkb: bytes | None) -> tuple[list[Line], str, dict[str, int]]:
    """Decode valid line parts without bulk GEOS construction.

    The decoder supports standard/EWKB Point, LineString, MultiLineString, and
    GeometryCollection records. Non-line primitives are counted and skipped.
    Malformed line parts never discard valid parts already decoded.
    """
    reasons = {
        "null_geometry": 0,
        "empty_geometry": 0,
        "non_line_geometry": 0,
        "fewer_than_two_coordinates": 0,
        "fewer_than_two_distinct_coordinates": 0,
        "nonfinite_coordinate": 0,
        "invalid_wkb": 0,
    }
    if raw_wkb is None:
        reasons["null_geometry"] = 1
        return [], "Null", reasons
    data = bytes(raw_wkb)
    if not data:
        reasons["empty_geometry"] = 1
        return [], "Empty", reasons

    type_names = {
        1: "Point",
        2: "LineString",
        3: "Polygon",
        4: "MultiPoint",
        5: "MultiLineString",
        6: "MultiPolygon",
        7: "GeometryCollection",
    }
    parts: list[Line] = []

    def skip_geometry(offset: int) -> int:
        endian, base_type, dimensions, cursor = _wkb_header(data, offset)
        if base_type == 1:
            _, _, cursor = _wkb_coordinates(data, cursor, endian, dimensions, 1)
            return cursor
        if base_type == 2:
            count, cursor = _wkb_uint32(data, cursor, endian)
            _, _, cursor = _wkb_coordinates(data, cursor, endian, dimensions, count)
            return cursor
        if base_type == 3:
            ring_count, cursor = _wkb_uint32(data, cursor, endian)
            for _ in range(ring_count):
                count, cursor = _wkb_uint32(data, cursor, endian)
                _, _, cursor = _wkb_coordinates(data, cursor, endian, dimensions, count)
            return cursor
        if base_type in {4, 5, 6, 7}:
            geometry_count, cursor = _wkb_uint32(data, cursor, endian)
            for _ in range(geometry_count):
                cursor = skip_geometry(cursor)
            return cursor
        raise WkbDecodeError(f"unsupported WKB geometry type {base_type}")

    def collect_geometry(offset: int) -> int:
        endian, base_type, dimensions, cursor = _wkb_header(data, offset)
        if base_type == 2:
            count, cursor = _wkb_uint32(data, cursor, endian)
            points, all_finite, cursor = _wkb_coordinates(data, cursor, endian, dimensions, count)
            if count == 0:
                reasons["empty_geometry"] += 1
            elif count < 2:
                reasons["fewer_than_two_coordinates"] += 1
            elif not all_finite:
                reasons["nonfinite_coordinate"] += 1
            elif len(set(points)) < 2:
                reasons["fewer_than_two_distinct_coordinates"] += 1
            else:
                parts.append(points)
            return cursor
        if base_type in {5, 7}:
            geometry_count, cursor = _wkb_uint32(data, cursor, endian)
            for _ in range(geometry_count):
                cursor = collect_geometry(cursor)
            return cursor
        reasons["non_line_geometry"] += 1
        return skip_geometry(offset)

    try:
        _, top_type, _, _ = _wkb_header(data, 0)
        top_type_name = type_names.get(top_type, f"Type{top_type}")
        collect_geometry(0)
    except (WkbDecodeError, struct.error, OverflowError, ValueError):
        reasons["invalid_wkb"] += 1
        top_type_name = "InvalidWKB"
    return parts, top_type_name, reasons


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def affine_cell_area_m2(
    pixel_width: float,
    row_rotation: float,
    column_rotation: float,
    pixel_height: float,
) -> float:
    """Return physical cell area from the 2-D affine determinant."""
    area = abs(pixel_width * pixel_height - row_rotation * column_rotation)
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Projected raster has a non-positive or non-finite cell area")
    return area


def source_area_to_cells(source_area_km2: float, cell_area_m2: float) -> int:
    """Convert a physical source-area threshold to a conservative cell count."""
    if not math.isfinite(source_area_km2) or source_area_km2 <= 0:
        raise ValueError("--source-area-km2 must be finite and greater than zero")
    if not math.isfinite(cell_area_m2) or cell_area_m2 <= 0:
        raise ValueError("cell_area_m2 must be finite and greater than zero")
    return max(1, int(math.ceil(source_area_km2 * 1_000_000.0 / cell_area_m2)))


def automatic_utm_epsg(longitude: float, latitude: float) -> int:
    """Choose a WGS84 UTM CRS for a mask centroid."""
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        raise ValueError("Longitude/latitude must be expressed in valid WGS84 degrees")
    if latitude < -80.0 or latitude > 84.0:
        raise ValueError("Automatic UTM is unavailable outside 80 S to 84 N; pass --target-crs")
    zone = min(60, max(1, int(math.floor((longitude + 180.0) / 6.0)) + 1))
    return (32600 if latitude >= 0 else 32700) + zone


def line_length(points: Sequence[Point]) -> float:
    return sum(
        math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
        for index in range(len(points) - 1)
    )


def node_key(point: Point, tolerance_m: float) -> tuple[int, int]:
    if tolerance_m <= 0:
        raise ValueError("node tolerance must be greater than zero")
    return (round(point[0] / tolerance_m), round(point[1] / tolerance_m))


def classify_channel(slope: float, meanmsq: float) -> tuple[int, float, float, float]:
    """Retain the proven DHSVM-PNNL channel-class lookup."""
    area_bins = [
        (1_000_000.0, 0.5),
        (10_000_000.0, 1.0),
        (20_000_000.0, 2.0),
        (30_000_000.0, 3.0),
        (40_000_000.0, 4.0),
        (float("inf"), 4.5),
    ]
    bin_index = 0
    base_width = 0.5
    for bin_index, (limit, width) in enumerate(area_bins):
        if meanmsq <= limit:
            base_width = width
            break
    if slope <= 0.002:
        class_base, hydraulic_depth, effective_widths = 1, 0.03, [0.06, 0.09, 0.12, 0.15, 0.18, 0.21]
    elif slope <= 0.1:
        class_base, hydraulic_depth, effective_widths = 7, 0.05, [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    else:
        class_base, hydraulic_depth, effective_widths = 13, 0.10, [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    return class_base + bin_index, base_width, hydraulic_depth, effective_widths[bin_index]


def _finite_sample(value: float) -> bool:
    return math.isfinite(value) and not math.isclose(value, NODATA)


def build_arc_records(
    lines: Iterable[Sequence[Point]],
    elevation_sampler: Sampler,
    accumulation_sampler: Sampler,
    *,
    cell_area_m2: float,
    node_tolerance_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Orient, connect, and order GRASS stream arcs.

    SegOrder is the DHSVM longest-upstream-path order: every headwater has order
    one and a downstream arc receives one plus the maximum order of all direct
    upstream arcs.
    """
    normalized = [
        [(float(x), float(y)) for x, y in line]
        for line in lines
        if len(line) >= 2 and line_length(line) > 0
    ]
    start_points = [line[0] for line in normalized]
    end_points = [line[-1] for line in normalized]
    start_elevations = list(elevation_sampler(start_points))
    end_elevations = list(elevation_sampler(end_points))
    start_accumulations = list(accumulation_sampler(start_points))
    end_accumulations = list(accumulation_sampler(end_points))
    expected = len(normalized)
    if not all(len(values) == expected for values in (start_elevations, end_elevations, start_accumulations, end_accumulations)):
        raise ValueError("Raster samplers must return exactly one value for every endpoint")

    oriented: list[dict[str, Any]] = []
    invalid_endpoint_elevation_count = 0
    for line, selev, eelev, sacc, eacc in zip(
        normalized,
        start_elevations,
        end_elevations,
        start_accumulations,
        end_accumulations,
    ):
        selev = float(selev)
        eelev = float(eelev)
        sacc = float(sacc)
        eacc = float(eacc)
        reverse = False
        if _finite_sample(selev) and _finite_sample(eelev):
            reverse = selev < eelev
        elif _finite_sample(sacc) and _finite_sample(eacc):
            reverse = sacc > eacc
            invalid_endpoint_elevation_count += 1
        else:
            invalid_endpoint_elevation_count += 1
        if reverse:
            line = list(reversed(line))
            selev, eelev = eelev, selev
            sacc, eacc = eacc, sacc
        oriented.append(
            {
                "points": list(line),
                "SELEV": selev if _finite_sample(selev) else NODATA,
                "EELEV": eelev if _finite_sample(eelev) else NODATA,
                "end_accumulation_cells": eacc if _finite_sample(eacc) else 0.0,
            }
        )

    # Stable geometry ordering makes IDs independent of source feature order.
    oriented.sort(
        key=lambda item: (
            node_key(item["points"][0], node_tolerance_m),
            node_key(item["points"][-1], node_tolerance_m),
            round(line_length(item["points"]), 6),
            tuple((round(x, 6), round(y, 6)) for x, y in item["points"]),
        )
    )

    node_ids: dict[tuple[int, int], int] = {}

    def get_node_id(point: Point) -> int:
        key = node_key(point, node_tolerance_m)
        if key not in node_ids:
            node_ids[key] = len(node_ids) + 1
        return node_ids[key]

    records: list[dict[str, Any]] = []
    for arcid, item in enumerate(oriented, start=1):
        points = item["points"]
        length = max(line_length(points), 0.001)
        selev = item["SELEV"]
        eelev = item["EELEV"]
        dz = max(0.0, selev - eelev) if _finite_sample(selev) and _finite_sample(eelev) else 0.0
        maxgrid = max(0, int(round(item["end_accumulation_cells"])))
        records.append(
            {
                "arcid": arcid,
                "points": points,
                "from_node": get_node_id(points[0]),
                "to_node": get_node_id(points[-1]),
                "SELEV": selev,
                "EELEV": eelev,
                "MAXGRID": maxgrid,
                "Shape_Leng": length,
                "dz": dz,
                "slope": max(dz / length, 0.00001),
            }
        )

    starts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ends: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        starts[record["from_node"]].append(record)
        ends[record["to_node"]].append(record)

    ambiguous_downstream_nodes: list[dict[str, Any]] = []
    for record in records:
        downstream = sorted(
            starts.get(record["to_node"], []),
            key=lambda candidate: (-candidate["MAXGRID"], candidate["arcid"]),
        )
        if len(downstream) > 1:
            ambiguous_downstream_nodes.append(
                {
                    "arcid": record["arcid"],
                    "node": record["to_node"],
                    "candidate_arcids": [candidate["arcid"] for candidate in downstream],
                }
            )
        upstream = ends.get(record["from_node"], [])
        record["downarc"] = downstream[0]["arcid"] if downstream else -1
        record["uparc"] = max(upstream, key=lambda candidate: (candidate["MAXGRID"], -candidate["arcid"]))["arcid"] if upstream else -1
        upstream_sum = sum(candidate["MAXGRID"] for candidate in upstream)
        record["local"] = max(0, record["MAXGRID"] - upstream_sum)
        record["meanmsq"] = (record["MAXGRID"] + record["local"] / 2.0) * cell_area_m2
        record["drainage_area_m2"] = record["MAXGRID"] * cell_area_m2

    by_id = {record["arcid"]: record for record in records}
    upstream_by_arc: dict[int, list[int]] = defaultdict(list)
    downstream_by_arc: dict[int, int] = {}
    for record in records:
        if record["downarc"] != -1:
            downstream_by_arc[record["arcid"]] = record["downarc"]
            upstream_by_arc[record["downarc"]].append(record["arcid"])

    queue = deque(sorted(record["arcid"] for record in records if not upstream_by_arc.get(record["arcid"])))
    for arcid in queue:
        by_id[arcid]["segorder"] = 1
    processed: set[int] = set()
    while queue:
        arcid = queue.popleft()
        processed.add(arcid)
        downarc = downstream_by_arc.get(arcid)
        if downarc is None:
            continue
        upstream_ids = upstream_by_arc.get(downarc, [])
        if all("segorder" in by_id[upstream_id] for upstream_id in upstream_ids):
            by_id[downarc]["segorder"] = 1 + max(by_id[upstream_id]["segorder"] for upstream_id in upstream_ids)
            if downarc not in processed:
                queue.append(downarc)

    for record in records:
        record.setdefault("segorder", -1)
        channel_class, hydraulic_width, hydraulic_depth, effective_width = classify_channel(
            record["slope"], record["meanmsq"]
        )
        record.update(
            {
                "chanclass": channel_class,
                "hyddepth": hydraulic_depth,
                "hydwidth": hydraulic_width,
                "effwidth": effective_width,
                "effdepth": NODATA,
                "segdepth": NODATA,
            }
        )

    invalid_references: list[list[int]] = []
    segorder_errors: list[list[int]] = []
    for record in records:
        for field in ("downarc", "uparc"):
            reference = record[field]
            if reference != -1 and reference not in by_id:
                invalid_references.append([record["arcid"], reference])
        downarc = record["downarc"]
        if downarc != -1 and by_id[downarc]["segorder"] <= record["segorder"]:
            segorder_errors.append([record["arcid"], downarc])

    unassigned = [record["arcid"] for record in records if record["segorder"] < 1]
    qa = {
        "arc_count": len(records),
        "node_count": len(node_ids),
        "headwater_segments": sum(1 for record in records if record["uparc"] == -1),
        "terminal_segments": sum(1 for record in records if record["downarc"] == -1),
        "multiple_terminal_segments_diagnostic": sum(1 for record in records if record["downarc"] == -1) > 1,
        "invalid_endpoint_elevation_count": invalid_endpoint_elevation_count,
        "ambiguous_downstream_nodes": ambiguous_downstream_nodes,
        "invalid_references": invalid_references,
        "unassigned_segorder_arcids": unassigned,
        "segorder_errors": segorder_errors,
        "has_cycle_or_unresolved_topology": bool(unassigned),
    }
    return records, qa


def network_dat_rows(records: Sequence[dict[str, Any]]) -> list[str]:
    return [
        (
            f"{record['arcid']:5d} {record['segorder']:3d} {record['slope']:11.5f} "
            f"{record['Shape_Leng']:17.5f} {record['chanclass']:3d} {record['downarc']:7d}"
        )
        for record in sorted(records, key=lambda item: item["arcid"])
    ]
