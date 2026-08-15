from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import netCDF4 as nc4
import numpy as np
from scipy.spatial import Delaunay, cKDTree


MJD_EPOCH_MS = int(np.datetime64("1858-11-17T00:00:00", "ms").astype(np.int64))
DAY_MS = 86_400_000
VALUE_ALIASES = ("elevation", "water_level", "waterlevel", "ssh", "zeta", "surf_el")
TIME_ALIASES = ("time", "Times", "datetime", "date_time", "timestamp")
LON_ALIASES = ("lon", "longitude", "xlon")
LAT_ALIASES = ("lat", "latitude", "ylat")
NODE_ALIASES = ("obc_nodes", "node_id", "node", "nodes")
STATION_ALIASES = ("station_id", "station", "site_id", "site")


@dataclass(frozen=True)
class Boundary:
    node_ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    arcs: tuple[np.ndarray, ...]
    source: str


@dataclass(frozen=True)
class SourceSeries:
    times_ms: np.ndarray
    values: np.ndarray
    layout: str
    units: str
    lon: np.ndarray | None = None
    lat: np.ndarray | None = None
    node_ids: np.ndarray | None = None
    source_variable: str = ""
    time_source: str = ""


def _first_present(names: Iterable[str], available: Iterable[str]) -> str | None:
    lookup = {name.lower(): name for name in available}
    for candidate in names:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _validate_geographic(lon: np.ndarray, lat: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError(f"{label} contains non-finite coordinates")
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 360.0):
        raise ValueError(
            f"{label} must use geographic longitude/latitude degrees; projected coordinates need conversion first"
        )


def read_boundary_2dm(path: str | Path, open_nodestrings: Iterable[int]) -> Boundary:
    mesh_path = Path(path)
    requested = [int(value) for value in open_nodestrings]
    if not requested:
        raise ValueError("At least one --open-ns value is required for a 2DM mesh")

    nodes: dict[int, tuple[float, float]] = {}
    nodestrings: dict[int, tuple[int, ...]] = {}
    pending: list[int] = []

    with mesh_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            parts = raw.strip().split()
            if not parts:
                continue
            record = parts[0].upper()
            if record == "ND":
                if len(parts) < 5:
                    raise ValueError(f"Malformed ND record at {mesh_path}:{line_number}")
                nodes[int(parts[1])] = (float(parts[2]), float(parts[3]))
            elif record == "NS":
                values = [int(value) for value in parts[1:]]
                ended = False
                for index, value in enumerate(values):
                    if value < 0:
                        pending.append(abs(value))
                        if index + 1 >= len(values):
                            raise ValueError(f"Nodestring id missing at {mesh_path}:{line_number}")
                        nodestring_id = int(values[index + 1])
                        if nodestring_id in nodestrings:
                            raise ValueError(f"Duplicate nodestring id {nodestring_id} in {mesh_path}")
                        nodestrings[nodestring_id] = tuple(pending)
                        pending = []
                        ended = True
                        break
                    pending.append(value)
                if ended:
                    continue

    if pending:
        raise ValueError(f"Unterminated nodestring in {mesh_path}")
    missing_ns = [value for value in requested if value not in nodestrings]
    if missing_ns:
        raise ValueError(f"Missing nodestrings {missing_ns}; available ids are {sorted(nodestrings)}")

    node_ids: list[int] = []
    arcs: list[np.ndarray] = []
    for nodestring_id in requested:
        arc_nodes = list(nodestrings[nodestring_id])
        absent = [node for node in arc_nodes if node not in nodes]
        if absent:
            raise ValueError(f"Nodestring {nodestring_id} references missing nodes {absent[:5]}")
        start = len(node_ids)
        node_ids.extend(arc_nodes)
        arcs.append(np.arange(start, len(node_ids), dtype=np.int64))

    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Selected open nodestrings contain duplicate node ids")
    lon = np.asarray([nodes[node][0] for node in node_ids], dtype=np.float64)
    lat = np.asarray([nodes[node][1] for node in node_ids], dtype=np.float64)
    _validate_geographic(lon, lat, "2DM open boundary")
    return Boundary(
        node_ids=np.asarray(node_ids, dtype=np.int64),
        lon=lon,
        lat=lat,
        arcs=tuple(arcs),
        source=str(mesh_path),
    )


def read_boundary_dat(grd_path: str | Path, obc_path: str | Path) -> Boundary:
    grid_path = Path(grd_path)
    lines = grid_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Incomplete FVCOM grid file: {grid_path}")
    try:
        node_count = int(lines[0].split("=")[-1])
        cell_count = int(lines[1].split("=")[-1])
    except Exception as exc:
        raise ValueError(f"Cannot parse node/cell counts in {grid_path}") from exc

    element_lines = lines[2 : 2 + cell_count]
    node_lines = lines[2 + cell_count : 2 + cell_count + node_count]
    if len(element_lines) != cell_count or len(node_lines) != node_count:
        raise ValueError(f"Grid counts do not match records in {grid_path}")

    edges: set[tuple[int, int]] = set()
    for raw in element_lines:
        parts = raw.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed element record in {grid_path}: {raw!r}")
        tri = [int(parts[1]), int(parts[2]), int(parts[3])]
        for left, right in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.add((min(left, right), max(left, right)))

    nodes: dict[int, tuple[float, float]] = {}
    for raw in node_lines:
        parts = raw.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed node record in {grid_path}: {raw!r}")
        nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))

    obc_lines = Path(obc_path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not obc_lines:
        raise ValueError(f"Empty FVCOM OBC file: {obc_path}")
    try:
        expected = int(obc_lines[0].split("=")[-1])
    except Exception as exc:
        raise ValueError(f"Cannot parse OBC count in {obc_path}") from exc
    node_ids = [int(raw.split()[1]) for raw in obc_lines[1:] if raw.strip()]
    if len(node_ids) != expected:
        raise ValueError(f"OBC count {expected} does not match {len(node_ids)} records")
    absent = [node for node in node_ids if node not in nodes]
    if absent:
        raise ValueError(f"OBC file references missing grid nodes {absent[:5]}")

    breaks = [0]
    for index, (left, right) in enumerate(zip(node_ids[:-1], node_ids[1:]), start=1):
        if (min(left, right), max(left, right)) not in edges:
            breaks.append(index)
    breaks.append(len(node_ids))
    arcs = tuple(
        np.arange(start, stop, dtype=np.int64)
        for start, stop in zip(breaks[:-1], breaks[1:])
        if stop > start
    )
    lon = np.asarray([nodes[node][0] for node in node_ids], dtype=np.float64)
    lat = np.asarray([nodes[node][1] for node in node_ids], dtype=np.float64)
    _validate_geographic(lon, lat, "FVCOM DAT open boundary")
    return Boundary(
        node_ids=np.asarray(node_ids, dtype=np.int64),
        lon=lon,
        lat=lat,
        arcs=arcs,
        source=f"{grid_path}|{Path(obc_path)}",
    )


def parse_iso_time_ms(text: str, assume_utc: bool = False) -> int:
    value = str(text).strip().replace("/", "-")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp {text!r}") from exc
    if parsed.tzinfo is None:
        if not assume_utc:
            raise ValueError(f"Timestamp {text!r} has no timezone; pass --assume-utc only when appropriate")
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return int(round(parsed.timestamp() * 1000.0))


def _decode_char_times(variable: Any) -> list[str]:
    raw = variable[:]
    if raw.ndim == 2:
        decoded = nc4.chartostring(raw)
    else:
        decoded = raw
    out: list[str] = []
    for value in np.asarray(decoded).ravel():
        if isinstance(value, bytes):
            out.append(value.decode("ascii", errors="strict").strip())
        else:
            out.append(str(value).strip())
    return out


def decode_netcdf_times(
    dataset: nc4.Dataset,
    override: str | None = None,
    assume_utc: bool = False,
) -> tuple[np.ndarray, str, str]:
    variables = dataset.variables
    if override:
        if override not in variables:
            raise ValueError(f"Requested time variable {override!r} is absent")
        if override == "Itime" and "Itime2" in variables:
            days = np.asarray(variables["Itime"][:], dtype=np.int64)
            millis = np.asarray(variables["Itime2"][:], dtype=np.int64)
            return MJD_EPOCH_MS + days * DAY_MS + millis, "Itime+Itime2", variables["Itime"].dimensions[0]
        candidates = [override]
    else:
        if "Times" in variables:
            candidates = ["Times"]
        elif "Itime" in variables and "Itime2" in variables:
            days = np.asarray(variables["Itime"][:], dtype=np.int64)
            millis = np.asarray(variables["Itime2"][:], dtype=np.int64)
            return MJD_EPOCH_MS + days * DAY_MS + millis, "Itime+Itime2", variables["Itime"].dimensions[0]
        else:
            name = _first_present(("time", "datetime", "timestamp"), variables)
            candidates = [name] if name else []

    if not candidates:
        raise ValueError("No supported NetCDF time representation found")
    name = candidates[0]
    variable = variables[name]
    time_dim = variable.dimensions[0]
    if variable.dtype.kind in {"S", "U"} or name == "Times":
        zone = str(getattr(variable, "time_zone", "")).upper()
        allow_naive = assume_utc or zone == "UTC"
        values = np.asarray([parse_iso_time_ms(item, allow_naive) for item in _decode_char_times(variable)])
        return values.astype(np.int64), name, time_dim

    units = getattr(variable, "units", None)
    if not units:
        raise ValueError(f"Numeric time variable {name!r} lacks CF/FVCOM units")
    calendar = str(getattr(variable, "calendar", "standard"))
    dates = nc4.num2date(
        variable[:],
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    output: list[int] = []
    for item in np.asarray(dates).ravel():
        microsecond = int(getattr(item, "microsecond", 0))
        parsed = datetime(
            int(item.year), int(item.month), int(item.day), int(item.hour), int(item.minute), int(item.second),
            microsecond, tzinfo=timezone.utc,
        )
        output.append(int(round(parsed.timestamp() * 1000.0)))
    return np.asarray(output, dtype=np.int64), name, time_dim


def unit_scale_to_metres(units: str) -> float:
    key = str(units).strip().lower().replace(" ", "")
    if key in {"m", "meter", "meters", "metre", "metres"}:
        return 1.0
    if key in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return 0.01
    if key in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return 0.001
    raise ValueError(f"Unsupported or ambiguous water-level units {units!r}; use m, cm, or mm")


def _read_coord(dataset: nc4.Dataset, override: str | None, aliases: tuple[str, ...]) -> tuple[str, Any] | None:
    name = override or _first_present(aliases, dataset.variables)
    if name is None:
        return None
    if name not in dataset.variables:
        raise ValueError(f"Requested coordinate variable {name!r} is absent")
    return name, dataset.variables[name]


def read_netcdf_source(
    path: str | Path,
    *,
    value_var: str | None = None,
    time_var: str | None = None,
    lon_var: str | None = None,
    lat_var: str | None = None,
    node_var: str | None = None,
    units_override: str | None = None,
    assume_utc: bool = False,
) -> SourceSeries:
    source_path = Path(path)
    with nc4.Dataset(source_path) as dataset:
        name = value_var or _first_present(VALUE_ALIASES, dataset.variables)
        if name is None or name not in dataset.variables:
            raise ValueError(f"Cannot identify a water-level variable in {source_path}")
        variable = dataset.variables[name]
        times_ms, time_source, time_dim = decode_netcdf_times(dataset, time_var, assume_utc)
        if time_dim not in variable.dimensions:
            raise ValueError(f"Water-level variable {name!r} does not use time dimension {time_dim!r}")

        array = np.ma.filled(variable[:], np.nan).astype(np.float64, copy=False)
        time_axis = variable.dimensions.index(time_dim)
        array = np.moveaxis(array, time_axis, 0)
        spatial_dims = list(variable.dimensions)
        spatial_dims.pop(time_axis)
        while array.ndim > 1 and 1 in array.shape[1:]:
            axis = 1 + list(array.shape[1:]).index(1)
            array = np.squeeze(array, axis=axis)
            spatial_dims.pop(axis - 1)
        if array.shape[0] != len(times_ms):
            raise ValueError("Time coordinate and water-level variable lengths differ")

        source_units = units_override or getattr(variable, "units", "")
        scale = unit_scale_to_metres(source_units)
        array *= scale
        spatial_shape = array.shape[1:]
        values = array.reshape((array.shape[0], int(np.prod(spatial_shape)) if spatial_shape else 1))

        node_coord = _read_coord(dataset, node_var, NODE_ALIASES)
        if node_coord is not None:
            node_values = np.asarray(node_coord[1][:]).reshape(-1)
            if node_values.size == values.shape[1]:
                return SourceSeries(
                    times_ms=times_ms,
                    values=values,
                    layout="boundary_nodes",
                    units="meters",
                    node_ids=node_values.astype(np.int64),
                    source_variable=name,
                    time_source=time_source,
                )

        lon_coord = _read_coord(dataset, lon_var, LON_ALIASES)
        lat_coord = _read_coord(dataset, lat_var, LAT_ALIASES)
        if lon_coord is None or lat_coord is None:
            if values.shape[1] == 1:
                return SourceSeries(times_ms, values, "single_series", "meters", source_variable=name, time_source=time_source)
            raise ValueError(
                "Multi-point NetCDF input needs node ids or longitude/latitude coordinates"
            )

        lon_name, lon_nc = lon_coord
        lat_name, lat_nc = lat_coord
        lon_data = np.ma.filled(lon_nc[:], np.nan).astype(np.float64)
        lat_data = np.ma.filled(lat_nc[:], np.nan).astype(np.float64)
        layout = "stations"
        if lon_data.ndim == 1 and lat_data.ndim == 1 and lon_nc.dimensions != lat_nc.dimensions:
            target_shape = tuple(spatial_shape)
            lon_shape = [1] * len(target_shape)
            lat_shape = [1] * len(target_shape)
            try:
                lon_axis = spatial_dims.index(lon_nc.dimensions[0])
                lat_axis = spatial_dims.index(lat_nc.dimensions[0])
            except ValueError as exc:
                raise ValueError("Longitude/latitude dimensions do not align with the water-level field") from exc
            lon_shape[lon_axis] = lon_data.size
            lat_shape[lat_axis] = lat_data.size
            lon_grid = np.broadcast_to(lon_data.reshape(lon_shape), target_shape)
            lat_grid = np.broadcast_to(lat_data.reshape(lat_shape), target_shape)
            lon_flat = lon_grid.reshape(-1)
            lat_flat = lat_grid.reshape(-1)
            layout = "grid"
        elif lon_data.shape == lat_data.shape and lon_data.size == values.shape[1]:
            lon_flat = lon_data.reshape(-1)
            lat_flat = lat_data.reshape(-1)
            if lon_data.ndim > 1:
                layout = "grid"
        else:
            raise ValueError(
                f"Coordinate shapes for {lon_name}/{lat_name} do not match water-level spatial shape {spatial_shape}"
            )
        _validate_geographic(lon_flat, lat_flat, "NetCDF source")
        return SourceSeries(
            times_ms=times_ms,
            values=values,
            layout=layout,
            units="meters",
            lon=lon_flat,
            lat=lat_flat,
            source_variable=name,
            time_source=time_source,
        )


def read_csv_source(
    path: str | Path,
    *,
    value_var: str | None = None,
    time_var: str | None = None,
    lon_var: str | None = None,
    lat_var: str | None = None,
    node_var: str | None = None,
    units_override: str | None = None,
    assume_utc: bool = False,
) -> SourceSeries:
    if not units_override:
        raise ValueError("CSV inputs require --units m, cm, or mm")
    scale = unit_scale_to_metres(units_override)
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV file has no header: {source_path}")
        fields = reader.fieldnames
        time_name = time_var or _first_present(TIME_ALIASES, fields)
        value_name = value_var or _first_present(VALUE_ALIASES, fields)
        if time_name is None or value_name is None:
            raise ValueError("CSV input needs identifiable time and water-level columns")
        node_name = node_var or _first_present(NODE_ALIASES, fields)
        lon_name = lon_var or _first_present(LON_ALIASES, fields)
        lat_name = lat_var or _first_present(LAT_ALIASES, fields)
        station_name = _first_present(STATION_ALIASES, fields)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV input is empty: {source_path}")

    records: list[tuple[int, Any, float, float | None, float | None]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            stamp = parse_iso_time_ms(row[time_name], assume_utc)
            value = float(row[value_name]) * scale
            if node_name:
                key: Any = int(row[node_name])
            elif station_name:
                key = row[station_name]
            elif lon_name and lat_name:
                key = (float(row[lon_name]), float(row[lat_name]))
            else:
                key = "single"
            lon = float(row[lon_name]) if lon_name else None
            lat = float(row[lat_name]) if lat_name else None
        except Exception as exc:
            raise ValueError(f"Invalid CSV record at row {row_number}") from exc
        records.append((stamp, key, value, lon, lat))

    times = sorted({record[0] for record in records})
    keys = list(dict.fromkeys(record[1] for record in records))
    time_index = {value: index for index, value in enumerate(times)}
    key_index = {value: index for index, value in enumerate(keys)}
    values = np.full((len(times), len(keys)), np.nan, dtype=np.float64)
    coordinates: dict[Any, tuple[float, float]] = {}
    for stamp, key, value, lon, lat in records:
        index = (time_index[stamp], key_index[key])
        if np.isfinite(values[index]):
            raise ValueError(f"Duplicate CSV value for time {stamp} and key {key!r}")
        values[index] = value
        if lon is not None and lat is not None:
            old = coordinates.setdefault(key, (lon, lat))
            if not np.allclose(old, (lon, lat), atol=1e-10):
                raise ValueError(f"Coordinates change for CSV station {key!r}")

    if node_name:
        return SourceSeries(
            np.asarray(times, dtype=np.int64), values, "boundary_nodes", "meters",
            node_ids=np.asarray(keys, dtype=np.int64), source_variable=value_name, time_source=time_name,
        )
    if keys == ["single"]:
        return SourceSeries(
            np.asarray(times, dtype=np.int64), values, "single_series", "meters",
            source_variable=value_name, time_source=time_name,
        )
    if len(coordinates) != len(keys):
        raise ValueError("Station CSV input needs longitude and latitude for every station")
    lon = np.asarray([coordinates[key][0] for key in keys], dtype=np.float64)
    lat = np.asarray([coordinates[key][1] for key in keys], dtype=np.float64)
    _validate_geographic(lon, lat, "CSV stations")
    return SourceSeries(
        np.asarray(times, dtype=np.int64), values, "stations", "meters",
        lon=lon, lat=lat, source_variable=value_name, time_source=time_name,
    )


def read_source(path: str | Path, **kwargs: Any) -> SourceSeries:
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".nc4", ".cdf"}:
        source = read_netcdf_source(path, **kwargs)
    elif suffix == ".csv":
        source = read_csv_source(path, **kwargs)
    else:
        raise ValueError(f"Unsupported source extension {suffix!r}; use NetCDF or CSV")
    if source.values.shape[0] != len(source.times_ms):
        raise ValueError("Source time/value dimensions differ")
    if len(source.times_ms) < 2:
        raise ValueError("At least two source timestamps are required")
    order = np.argsort(source.times_ms, kind="stable")
    if not np.array_equal(order, np.arange(len(order))):
        source = SourceSeries(
            source.times_ms[order], source.values[order], source.layout, source.units,
            source.lon, source.lat, source.node_ids, source.source_variable, source.time_source,
        )
    if np.any(np.diff(source.times_ms) <= 0):
        raise ValueError("Source timestamps must be unique and strictly increasing")
    return source


def _local_xy_km(lon: np.ndarray, lat: np.ndarray, center_lon: float, center_lat: float) -> np.ndarray:
    delta_lon = (np.asarray(lon, dtype=np.float64) - center_lon + 180.0) % 360.0 - 180.0
    x = delta_lon * 111.320 * math.cos(math.radians(center_lat))
    y = (np.asarray(lat, dtype=np.float64) - center_lat) * 110.574
    return np.column_stack((x, y))


def _source_spacing_km(points: np.ndarray) -> float:
    if len(points) < 2:
        return math.inf
    distances, _ = cKDTree(points).query(points, k=2)
    finite = distances[:, 1][np.isfinite(distances[:, 1]) & (distances[:, 1] > 0)]
    return float(np.median(finite)) if finite.size else math.inf


def spatial_interpolate(
    source: SourceSeries,
    boundary: Boundary,
    *,
    broadcast_single: bool = False,
    station_power: float = 2.0,
    max_nearest_km: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if source.layout == "boundary_nodes":
        if source.node_ids is None:
            raise ValueError("Boundary-node source lacks node ids")
        index = {int(node): position for position, node in enumerate(source.node_ids)}
        missing = [int(node) for node in boundary.node_ids if int(node) not in index]
        if missing:
            raise ValueError(f"Source is missing FVCOM boundary nodes {missing[:10]}")
        order = np.asarray([index[int(node)] for node in boundary.node_ids], dtype=np.int64)
        return source.values[:, order].copy(), {
            "method": "direct_node_id",
            "source_point_count": int(source.values.shape[1]),
            "fallback_target_count": 0,
        }

    if source.layout == "single_series":
        if not broadcast_single:
            raise ValueError(
                "A single water-level series has no spatial information; pass --broadcast-single-series explicitly"
            )
        return np.repeat(source.values[:, :1], len(boundary.node_ids), axis=1), {
            "method": "explicit_single_series_broadcast",
            "source_point_count": 1,
            "fallback_target_count": 0,
        }

    if source.lon is None or source.lat is None:
        raise ValueError("Spatial source lacks longitude/latitude coordinates")
    coordinate_ok = np.isfinite(source.lon) & np.isfinite(source.lat)
    if coordinate_ok.sum() < 2:
        raise ValueError("At least two finite spatial source coordinates are required")
    src_lon = source.lon[coordinate_ok]
    src_lat = source.lat[coordinate_ok]
    values = source.values[:, coordinate_ok]
    center_lon = float(np.angle(np.mean(np.exp(1j * np.radians(src_lon))), deg=True))
    center_lat = float(np.mean(np.concatenate((src_lat, boundary.lat))))
    source_points = _local_xy_km(src_lon, src_lat, center_lon, center_lat)
    target_points = _local_xy_km(boundary.lon, boundary.lat, center_lon, center_lat)
    spacing = _source_spacing_km(source_points)
    limit = float(max_nearest_km) if max_nearest_km is not None else 2.0 * spacing
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Cannot infer a bounded nearest-neighbour distance; provide --max-nearest-km")

    tree = cKDTree(source_points)
    nearest_distance, nearest_index = tree.query(target_points, k=min(8, len(source_points)))
    nearest_distance = np.atleast_2d(nearest_distance)
    nearest_index = np.atleast_2d(nearest_index)
    if nearest_distance.shape[0] != len(target_points):
        nearest_distance = nearest_distance.T
        nearest_index = nearest_index.T
    closest = nearest_distance[:, 0]
    if np.any(closest > limit):
        bad = np.where(closest > limit)[0]
        raise ValueError(
            f"{len(bad)} boundary nodes exceed nearest-source limit {limit:.3f} km; maximum is {closest.max():.3f} km"
        )

    output = np.full((values.shape[0], len(boundary.node_ids)), np.nan, dtype=np.float64)
    fallback_targets: set[int] = set()
    if source.layout == "stations":
        count = min(4, len(source_points))
        distances, indices = tree.query(target_points, k=count)
        distances = np.asarray(distances)
        indices = np.asarray(indices)
        if count == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        weights = 1.0 / np.maximum(distances, 1e-9) ** float(station_power)
        exact = distances[:, 0] < 1e-9
        for target in range(len(target_points)):
            if exact[target]:
                output[:, target] = values[:, indices[target, 0]]
                continue
            block = values[:, indices[target]]
            valid = np.isfinite(block)
            weighted = np.where(valid, weights[target][None, :], 0.0)
            denominator = weighted.sum(axis=1)
            numerator = np.nansum(block * weighted, axis=1)
            output[:, target] = np.divide(
                numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
            )
        method = "station_idw"
    else:
        if len(source_points) < 3:
            raise ValueError("Gridded/curvilinear interpolation needs at least three finite coordinates")
        triangulation = Delaunay(source_points)
        simplex = triangulation.find_simplex(target_points)
        for target, simplex_id in enumerate(simplex):
            if simplex_id < 0:
                fallback_targets.add(target)
                continue
            transform = triangulation.transform[simplex_id]
            bary = np.dot(transform[:2], target_points[target] - transform[2])
            weights = np.append(bary, 1.0 - bary.sum())
            vertices = triangulation.simplices[simplex_id]
            block = values[:, vertices]
            valid = np.all(np.isfinite(block), axis=1)
            output[valid, target] = block[valid] @ weights
            if not np.all(valid):
                fallback_targets.add(target)
        method = "triangulated_linear_with_nearest_wet_fallback"

    missing_rows, missing_targets = np.where(~np.isfinite(output))
    for row, target in zip(missing_rows.tolist(), missing_targets.tolist()):
        filled = False
        for distance, source_index in zip(nearest_distance[target], nearest_index[target]):
            if distance <= limit and np.isfinite(values[row, source_index]):
                output[row, target] = values[row, source_index]
                fallback_targets.add(target)
                filled = True
                break
        if not filled:
            continue

    return output, {
        "method": method,
        "source_point_count": int(len(source_points)),
        "fallback_target_count": int(len(fallback_targets)),
        "fallback_target_indices": sorted(int(value) for value in fallback_targets),
        "source_median_spacing_km": float(spacing),
        "nearest_limit_km": float(limit),
        "maximum_nearest_distance_km": float(np.max(closest)),
    }


def build_target_times(
    source_times_ms: np.ndarray,
    *,
    start: str | None = None,
    end: str | None = None,
    dt_seconds: float | None = None,
    assume_utc: bool = False,
) -> np.ndarray:
    provided = [start is not None, end is not None, dt_seconds is not None]
    if any(provided) and not all(provided):
        raise ValueError("Provide --start, --end, and --dt-seconds together")
    if not any(provided):
        differences = np.diff(source_times_ms)
        if np.max(np.abs(differences - int(np.median(differences)))) > 1:
            raise ValueError("Irregular source timestamps require an explicit target time grid")
        return source_times_ms.copy()

    assert start is not None and end is not None and dt_seconds is not None
    start_ms = parse_iso_time_ms(start, assume_utc)
    end_ms = parse_iso_time_ms(end, assume_utc)
    step_ms = int(round(float(dt_seconds) * 1000.0))
    if step_ms <= 0 or end_ms < start_ms:
        raise ValueError("Target timestep must be positive and end must not precede start")
    span = end_ms - start_ms
    if span % step_ms:
        raise ValueError("Target end must fall exactly on the requested timestep")
    return start_ms + np.arange(span // step_ms + 1, dtype=np.int64) * step_ms


def temporal_interpolate(
    source_times_ms: np.ndarray,
    values: np.ndarray,
    target_times_ms: np.ndarray,
    *,
    max_gap_factor: float = 3.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_gap_factor <= 1.0:
        raise ValueError("--max-gap-factor must exceed 1")
    if target_times_ms[0] < source_times_ms[0] or target_times_ms[-1] > source_times_ms[-1]:
        raise ValueError("Target period extends outside source coverage; extrapolation is disabled")
    source_diff = np.diff(source_times_ms)
    native_ms = int(np.median(source_diff))
    large = source_diff > max_gap_factor * native_ms
    if np.any(large):
        first = int(np.where(large)[0][0])
        raise ValueError(
            f"Source time gap {source_diff[first] / 3_600_000:.3f} h exceeds "
            f"{max_gap_factor:g} times native cadence"
        )
    if np.array_equal(source_times_ms, target_times_ms) and np.all(np.isfinite(values)):
        return values.copy(), {
            "method": "preserved_source_axis",
            "source_cadence_seconds": native_ms / 1000.0,
            "target_cadence_seconds": native_ms / 1000.0,
        }

    output = np.full((len(target_times_ms), values.shape[1]), np.nan, dtype=np.float64)
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column])
        if valid.sum() < 2:
            continue
        valid_times = source_times_ms[valid]
        valid_values = values[valid, column]
        output[:, column] = np.interp(target_times_ms, valid_times, valid_values)
        locations = np.searchsorted(valid_times, target_times_ms, side="left")
        left = np.clip(locations - 1, 0, len(valid_times) - 1)
        right = np.clip(locations, 0, len(valid_times) - 1)
        bracket = valid_times[right] - valid_times[left]
        invalid = bracket > max_gap_factor * native_ms
        output[invalid, column] = np.nan
    if not np.all(np.isfinite(output)):
        count = int(np.size(output) - np.isfinite(output).sum())
        raise ValueError(f"Temporal interpolation left {count} missing boundary values")
    target_cadence = int(np.median(np.diff(target_times_ms))) if len(target_times_ms) > 1 else 0
    return output, {
        "method": "linear_no_extrapolation",
        "source_cadence_seconds": native_ms / 1000.0,
        "target_cadence_seconds": target_cadence / 1000.0,
        "max_gap_factor": float(max_gap_factor),
    }


def format_times(times_ms: np.ndarray) -> np.ndarray:
    output = np.zeros((len(times_ms), 26), dtype="S1")
    for index, value in enumerate(times_ms):
        stamp = str(np.datetime64(int(value), "ms")).replace("T", " ")
        if "." not in stamp:
            stamp += ".000"
        date, fraction = stamp.split(".", 1)
        text = f"{date.replace('-', '/')}.{fraction[:3].ljust(3, '0')}000"
        output[index, :] = np.asarray(list(text[:26]), dtype="S1")
    return output


def write_fvcom_forcing(
    path: str | Path,
    boundary: Boundary,
    times_ms: np.ndarray,
    elevation_m: np.ndarray,
    *,
    case_name: str,
    vertical_datum: str,
    source_name: str,
    source_sha256: str,
    spatial_method: str,
) -> None:
    output = Path(path)
    if elevation_m.shape != (len(times_ms), len(boundary.node_ids)):
        raise ValueError("Elevation must have shape (time, nobc)")
    if not np.all(np.isfinite(elevation_m)):
        raise ValueError("Elevation contains non-finite values")
    if np.any(np.diff(times_ms) <= 0):
        raise ValueError("Output times must be strictly increasing")
    if len(times_ms) > np.iinfo(np.int32).max:
        raise ValueError("Too many output records for FVCOM iint int32")

    delta_ms = times_ms.astype(np.int64) - MJD_EPOCH_MS
    itime = np.floor_divide(delta_ms, DAY_MS).astype(np.int32)
    itime2 = np.mod(delta_ms, DAY_MS).astype(np.int32)
    time_mjd = (itime.astype(np.float64) + itime2.astype(np.float64) / DAY_MS).astype(np.float32)
    time_strings = format_times(times_ms)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        with nc4.Dataset(temp_name, "w", format="NETCDF3_CLASSIC") as dataset:
            dataset.type = "FVCOM TIME SERIES ELEVATION FORCING FILE"
            dataset.title = str(case_name)
            dataset.history = "Created by fvcom-boundary-waterlevel-forcing"
            dataset.source_file = Path(source_name).name
            dataset.source_sha256 = source_sha256
            dataset.vertical_datum = vertical_datum or "unspecified"
            dataset.spatial_interpolation = spatial_method

            dataset.createDimension("nobc", len(boundary.node_ids))
            dataset.createDimension("time", None)
            dataset.createDimension("DateStrLen", 26)

            variable = dataset.createVariable("obc_nodes", "i4", ("nobc",))
            variable.long_name = "Open Boundary Node Number"
            variable.grid = "obc_grid"
            variable[:] = boundary.node_ids.astype(np.int32)

            variable = dataset.createVariable("iint", "i4", ("time",))
            variable.long_name = "internal mode iteration number"
            variable[:] = np.arange(1, len(times_ms) + 1, dtype=np.int32)

            variable = dataset.createVariable("time", "f4", ("time",))
            variable.long_name = "time"
            variable.units = "days since 1858-11-17 00:00:00"
            variable.format = "modified julian day (MJD)"
            variable.time_zone = "UTC"
            variable[:] = time_mjd

            variable = dataset.createVariable("Itime", "i4", ("time",))
            variable.units = "days since 1858-11-17 00:00:00"
            variable.format = "modified julian day (MJD)"
            variable.time_zone = "UTC"
            variable[:] = itime

            variable = dataset.createVariable("Itime2", "i4", ("time",))
            variable.units = "msec since 00:00:00"
            variable.time_zone = "UTC"
            variable[:] = itime2

            variable = dataset.createVariable("Times", "S1", ("time", "DateStrLen"))
            variable.time_zone = "UTC"
            variable[:] = time_strings

            variable = dataset.createVariable("elevation", "f4", ("time", "nobc"))
            variable.long_name = "Open Boundary Elevation"
            variable.units = "meters"
            variable[:] = elevation_m.astype(np.float32)
        os.replace(temp_name, output)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def iso_utc(times_ms: np.ndarray) -> list[str]:
    return [str(np.datetime64(int(value), "ms")) + "Z" for value in times_ms]


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    value = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(value))


def arc_distances_km(boundary: Boundary, arc: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(arc), dtype=np.float64)
    for index in range(1, len(arc)):
        left = int(arc[index - 1])
        right = int(arc[index])
        distance[index] = distance[index - 1] + haversine_km(
            boundary.lon[left], boundary.lat[left], boundary.lon[right], boundary.lat[right]
        )
    return distance
