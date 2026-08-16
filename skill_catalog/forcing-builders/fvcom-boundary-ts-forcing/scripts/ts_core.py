from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import netCDF4 as nc4
import numpy as np


MJD_EPOCH_MS = int(np.datetime64("1858-11-17T00:00:00", "ms").astype(np.int64))
DAY_MS = 86_400_000
TEMP_ALIASES = ("obc_temp", "temperature", "temp", "water_temp", "thetao")
SALT_ALIASES = ("obc_salinity", "salinity", "salt", "so")
NODE_ALIASES = ("obc_nodes", "node_id", "node", "nodes")
SIGLAY_ALIASES = ("siglay", "sigma_layer", "sigma_layers")
SIGLEV_ALIASES = ("siglev", "sigma_level", "sigma_levels")


@dataclass(frozen=True)
class Boundary:
    node_ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    depth_m: np.ndarray
    arcs: tuple[np.ndarray, ...]
    source: str


@dataclass(frozen=True)
class SourceTS:
    times_ms: np.ndarray
    temperature_c: np.ndarray
    salinity: np.ndarray
    node_ids: np.ndarray
    siglay: np.ndarray
    siglev: np.ndarray
    source_variables: dict[str, str]
    source_units: dict[str, str]
    time_source: str
    extra_node_ids: np.ndarray
    sigma_orientation: str


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
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_geographic(lon: np.ndarray, lat: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError(f"{label} contains non-finite coordinates")
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 360.0):
        raise ValueError(f"{label} must use geographic longitude/latitude degrees")


def _validate_depth(depth: np.ndarray, label: str) -> None:
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError(f"{label} depth must be finite, positive-down metres")


def read_boundary_2dm(path: str | Path, open_nodestrings: Iterable[int]) -> Boundary:
    mesh_path = Path(path)
    requested = [int(value) for value in open_nodestrings]
    if not requested:
        raise ValueError("At least one --open-ns value is required for a 2DM mesh")
    nodes: dict[int, tuple[float, float, float]] = {}
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
                nodes[int(parts[1])] = (float(parts[2]), float(parts[3]), float(parts[4]))
            elif record == "NS":
                values = [int(value) for value in parts[1:]]
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
                        break
                    pending.append(value)
    if pending:
        raise ValueError(f"Unterminated nodestring in {mesh_path}")
    missing = [value for value in requested if value not in nodestrings]
    if missing:
        raise ValueError(f"Missing nodestrings {missing}; available ids are {sorted(nodestrings)}")
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
    depth = np.asarray([nodes[node][2] for node in node_ids], dtype=np.float64)
    _validate_geographic(lon, lat, "2DM open boundary")
    _validate_depth(depth, "2DM open boundary")
    return Boundary(np.asarray(node_ids, dtype=np.int64), lon, lat, depth, tuple(arcs), str(mesh_path))


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
    nodes: dict[int, tuple[float, float, float]] = {}
    for raw in node_lines:
        parts = raw.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed node record in {grid_path}: {raw!r}")
        nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
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
    arcs = tuple(np.arange(start, stop, dtype=np.int64) for start, stop in zip(breaks[:-1], breaks[1:]) if stop > start)
    lon = np.asarray([nodes[node][0] for node in node_ids], dtype=np.float64)
    lat = np.asarray([nodes[node][1] for node in node_ids], dtype=np.float64)
    depth = np.asarray([nodes[node][2] for node in node_ids], dtype=np.float64)
    _validate_geographic(lon, lat, "FVCOM DAT open boundary")
    _validate_depth(depth, "FVCOM DAT open boundary")
    return Boundary(np.asarray(node_ids, dtype=np.int64), lon, lat, depth, arcs, f"{grid_path}|{Path(obc_path)}")


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
            raise ValueError(f"Timestamp {text!r} has no timezone; pass --assume-utc only after confirming UTC")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(round(parsed.astimezone(timezone.utc).timestamp() * 1000.0))


def _decode_char_times(variable: Any) -> list[str]:
    raw = variable[:]
    decoded = nc4.chartostring(raw) if raw.ndim == 2 else raw
    return [value.decode("ascii").strip() if isinstance(value, bytes) else str(value).strip() for value in np.asarray(decoded).ravel()]


def decode_netcdf_times(dataset: nc4.Dataset, override: str | None = None, assume_utc: bool = False) -> tuple[np.ndarray, str, str]:
    variables = dataset.variables
    if override:
        if override not in variables:
            raise ValueError(f"Requested time variable {override!r} is absent")
        candidates = [override]
    elif "Times" in variables:
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
    if name == "Itime" and "Itime2" in variables:
        days = np.asarray(variables["Itime"][:], dtype=np.int64)
        millis = np.asarray(variables["Itime2"][:], dtype=np.int64)
        return MJD_EPOCH_MS + days * DAY_MS + millis, "Itime+Itime2", variables["Itime"].dimensions[0]
    variable = variables[name]
    time_dim = variable.dimensions[0]
    if variable.dtype.kind in {"S", "U"} or name == "Times":
        allow_naive = assume_utc or str(getattr(variable, "time_zone", "")).upper() == "UTC"
        values = [parse_iso_time_ms(item, allow_naive) for item in _decode_char_times(variable)]
        return np.asarray(values, dtype=np.int64), name, time_dim
    units = getattr(variable, "units", None)
    if not units:
        raise ValueError(f"Numeric time variable {name!r} lacks CF/FVCOM units")
    dates = nc4.num2date(variable[:], units=units, calendar=str(getattr(variable, "calendar", "standard")), only_use_cftime_datetimes=False, only_use_python_datetimes=False)
    output: list[int] = []
    for item in np.asarray(dates).ravel():
        parsed = datetime(int(item.year), int(item.month), int(item.day), int(item.hour), int(item.minute), int(item.second), int(getattr(item, "microsecond", 0)), tzinfo=timezone.utc)
        output.append(int(round(parsed.timestamp() * 1000.0)))
    return np.asarray(output, dtype=np.int64), name, time_dim


def _temperature_to_celsius(values: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    key = str(units).strip().lower().replace("°", "deg").replace(" ", "").replace("_", "")
    if key in {"c", "degc", "degreec", "degreesc", "celsius", "celcius", "degreecelsius", "degreescelsius"}:
        return values.astype(np.float64), "Celsius"
    if key in {"k", "kelvin", "degreek", "degreesk", "degreekelvin", "degreeskelvin"}:
        return values.astype(np.float64) - 273.15, "Celsius"
    raise ValueError(f"Unsupported or ambiguous temperature units {units!r}; use Celsius or Kelvin")


def _normalize_salinity(values: np.ndarray, units: str) -> tuple[np.ndarray, str]:
    key = str(units).strip().lower().replace(" ", "").replace("_", "")
    if key in {"psu", "1", "1e-3", "dimensionless", "practicalsalinityunit", "practicalsalinityunits", "ppt", "g/kg", "gkg-1"}:
        return values.astype(np.float64), "PSU"
    raise ValueError(f"Unsupported or ambiguous salinity units {units!r}; use PSU or a recognized dimensionless practical-salinity unit")


def _filled(variable: Any) -> np.ndarray:
    return np.ma.filled(variable[:], np.nan).astype(np.float64)


def _normalise_field(variable: Any, time_dim: str, layer_dim: str, node_dim: str) -> np.ndarray:
    dimensions = list(variable.dimensions)
    required = [time_dim, layer_dim, node_dim]
    if variable.ndim != 3 or any(name not in dimensions for name in required) or len(set(required)) != 3:
        raise ValueError(f"Variable {variable.name!r} must have dimensions equivalent to (time, siglay, node); found {tuple(dimensions)}")
    array = _filled(variable)
    return np.moveaxis(array, [dimensions.index(name) for name in required], [0, 1, 2])


def _normalise_sigma(variable: Any, level_dim: str, node_dim: str, source_node_count: int, reorder: np.ndarray) -> np.ndarray:
    array = _filled(variable)
    dimensions = list(variable.dimensions)
    if array.ndim == 1:
        if dimensions[0] != level_dim:
            raise ValueError(f"Sigma variable {variable.name!r} has unexpected dimension {dimensions}")
        return np.repeat(array[:, None], len(reorder), axis=1)
    if array.ndim != 2 or level_dim not in dimensions or node_dim not in dimensions:
        raise ValueError(f"Sigma variable {variable.name!r} must be 1-D or have (sigma, node) dimensions")
    normal = np.moveaxis(array, [dimensions.index(level_dim), dimensions.index(node_dim)], [0, 1])
    if normal.shape[1] != source_node_count:
        raise ValueError(f"Sigma variable {variable.name!r} node dimension does not match node IDs")
    return normal[:, reorder]


def validate_sigma(siglay: np.ndarray, siglev: np.ndarray, tolerance: float = 1.0e-3) -> str:
    if siglay.ndim != 2 or siglev.ndim != 2 or siglay.shape[1] != siglev.shape[1]:
        raise ValueError("siglay and siglev must be two-dimensional with the same node count")
    if siglev.shape[0] != siglay.shape[0] + 1:
        raise ValueError("siglev length must equal siglay length plus one")
    if not np.all(np.isfinite(siglay)) or not np.all(np.isfinite(siglev)):
        raise ValueError("Sigma coordinates contain missing or non-finite values")
    if np.any(siglay < -1.0 - tolerance) or np.any(siglay > tolerance) or np.any(siglev < -1.0 - tolerance) or np.any(siglev > tolerance):
        raise ValueError("Sigma coordinates must lie in [-1, 0]")
    orientations: list[str] = []
    for node in range(siglay.shape[1]):
        differences = np.diff(siglev[:, node])
        if np.all(differences > 0):
            orientation = "bottom_to_surface"
        elif np.all(differences < 0):
            orientation = "surface_to_bottom"
        else:
            raise ValueError(f"siglev is not strictly monotonic at boundary position {node}")
        orientations.append(orientation)
        endpoints = {round(float(siglev[0, node]), 6), round(float(siglev[-1, node]), 6)}
        if not (any(abs(value) <= tolerance for value in endpoints) and any(abs(value + 1.0) <= tolerance for value in endpoints)):
            raise ValueError(f"siglev endpoints must include 0 and -1 at boundary position {node}")
        lower = np.minimum(siglev[:-1, node], siglev[1:, node])
        upper = np.maximum(siglev[:-1, node], siglev[1:, node])
        if np.any(siglay[:, node] <= lower) or np.any(siglay[:, node] >= upper):
            raise ValueError(f"siglay values must fall strictly between adjacent siglev interfaces at boundary position {node}")
    if len(set(orientations)) != 1:
        raise ValueError("Sigma orientation differs between boundary nodes")
    return orientations[0]


def read_sigma_ready_source(
    path: str | Path,
    boundary: Boundary,
    *,
    temp_var: str | None = None,
    salt_var: str | None = None,
    time_var: str | None = None,
    node_var: str | None = None,
    siglay_var: str | None = None,
    siglev_var: str | None = None,
    temp_units: str | None = None,
    salt_units: str | None = None,
    assume_utc: bool = False,
) -> SourceTS:
    source_path = Path(path)
    if source_path.suffix.lower() not in {".nc", ".nc3", ".nc4", ".cdf"}:
        raise ValueError("The boundary T/S source must be NetCDF")
    with nc4.Dataset(source_path) as dataset:
        variables = dataset.variables
        names = {
            "temperature": temp_var or _first_present(TEMP_ALIASES, variables),
            "salinity": salt_var or _first_present(SALT_ALIASES, variables),
            "nodes": node_var or _first_present(NODE_ALIASES, variables),
            "siglay": siglay_var or _first_present(SIGLAY_ALIASES, variables),
            "siglev": siglev_var or _first_present(SIGLEV_ALIASES, variables),
        }
        absent = [key for key, value in names.items() if not value or value not in variables]
        if absent:
            raise ValueError(f"Missing required source variables: {absent}; use explicit variable-name overrides")
        times_ms, time_source, time_dim = decode_netcdf_times(dataset, time_var, assume_utc)
        node_variable = variables[names["nodes"]]
        if node_variable.ndim != 1:
            raise ValueError("Boundary node IDs must be one-dimensional")
        source_nodes = np.asarray(node_variable[:], dtype=np.int64)
        if len(np.unique(source_nodes)) != len(source_nodes):
            raise ValueError("Source boundary node IDs contain duplicates")
        lookup = {int(value): index for index, value in enumerate(source_nodes)}
        missing = [int(value) for value in boundary.node_ids if int(value) not in lookup]
        if missing:
            raise ValueError(f"Source is missing required FVCOM boundary node IDs {missing[:10]}")
        reorder = np.asarray([lookup[int(value)] for value in boundary.node_ids], dtype=np.int64)
        target_set = {int(value) for value in boundary.node_ids}
        extra = np.asarray([value for value in source_nodes if int(value) not in target_set], dtype=np.int64)
        node_dim = node_variable.dimensions[0]
        temperature_variable = variables[names["temperature"]]
        remaining = [dim for dim in temperature_variable.dimensions if dim not in {time_dim, node_dim}]
        if len(remaining) != 1:
            raise ValueError(f"Cannot identify the sigma-layer dimension of {names['temperature']!r}")
        layer_dim = remaining[0]
        temperature = _normalise_field(temperature_variable, time_dim, layer_dim, node_dim)[:, :, reorder]
        salinity = _normalise_field(variables[names["salinity"]], time_dim, layer_dim, node_dim)[:, :, reorder]
        if temperature.shape != salinity.shape:
            raise ValueError(f"Temperature and salinity shapes differ: {temperature.shape} versus {salinity.shape}")
        if temperature.shape[0] != len(times_ms):
            raise ValueError("T/S time dimension does not match decoded timestamps")
        siglay = _normalise_sigma(variables[names["siglay"]], layer_dim, node_dim, len(source_nodes), reorder)
        siglev_variable = variables[names["siglev"]]
        siglev_dims = [dim for dim in siglev_variable.dimensions if dim != node_dim]
        if len(siglev_dims) != 1:
            raise ValueError("Cannot identify the sigma-interface dimension")
        siglev = _normalise_sigma(siglev_variable, siglev_dims[0], node_dim, len(source_nodes), reorder)
        actual_temp_units = temp_units or str(getattr(temperature_variable, "units", ""))
        actual_salt_units = salt_units or str(getattr(variables[names["salinity"]], "units", ""))
        temperature, normalized_temp_units = _temperature_to_celsius(temperature, actual_temp_units)
        salinity, normalized_salt_units = _normalize_salinity(salinity, actual_salt_units)
    if len(times_ms) < 2 or np.any(np.diff(times_ms) <= 0):
        raise ValueError("Source timestamps must contain at least two strictly increasing records")
    orientation = validate_sigma(siglay, siglev)
    return SourceTS(
        times_ms=times_ms,
        temperature_c=temperature,
        salinity=salinity,
        node_ids=boundary.node_ids.copy(),
        siglay=siglay,
        siglev=siglev,
        source_variables={key: str(value) for key, value in names.items()},
        source_units={"temperature_input": actual_temp_units, "salinity_input": actual_salt_units, "temperature_output": normalized_temp_units, "salinity_output": normalized_salt_units},
        time_source=time_source,
        extra_node_ids=extra,
        sigma_orientation=orientation,
    )


def build_target_times(source_times_ms: np.ndarray, *, start: str | None = None, end: str | None = None, dt_seconds: float | None = None, assume_utc: bool = False) -> np.ndarray:
    provided = [start is not None, end is not None, dt_seconds is not None]
    if any(provided) and not all(provided):
        raise ValueError("Provide --start, --end, and --dt-seconds together")
    if not any(provided):
        differences = np.diff(source_times_ms)
        cadence = int(np.median(differences))
        if np.max(np.abs(differences - cadence)) > 1:
            raise ValueError("Irregular source timestamps require an explicit target time grid")
        return source_times_ms.copy()
    assert start is not None and end is not None and dt_seconds is not None
    start_ms = parse_iso_time_ms(start, assume_utc)
    end_ms = parse_iso_time_ms(end, assume_utc)
    step_ms = int(round(float(dt_seconds) * 1000.0))
    if step_ms <= 0 or end_ms <= start_ms:
        raise ValueError("Target timestep must be positive and end must follow start")
    if start_ms < source_times_ms[0] or end_ms > source_times_ms[-1]:
        raise ValueError("Target period extends outside source coverage; extrapolation is disabled")
    span = end_ms - start_ms
    if span % step_ms:
        raise ValueError("Target end must fall exactly on the requested timestep")
    return start_ms + np.arange(span // step_ms + 1, dtype=np.int64) * step_ms


def _max_consecutive(mask: np.ndarray) -> int:
    maximum = 0
    current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        maximum = max(maximum, current)
    return maximum


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = phi2 - phi1
    dlambda = np.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    value = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius * np.arcsin(np.sqrt(value)))


def arc_distances_km(boundary: Boundary, arc: np.ndarray) -> np.ndarray:
    distances = np.zeros(len(arc), dtype=np.float64)
    for index in range(1, len(arc)):
        left, right = int(arc[index - 1]), int(arc[index])
        distances[index] = distances[index - 1] + haversine_km(boundary.lon[left], boundary.lat[left], boundary.lon[right], boundary.lat[right])
    return distances


def _fill_line(line: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    missing = ~np.isfinite(line)
    valid = ~missing
    if not np.any(missing) or not np.any(valid):
        return np.zeros(line.shape, dtype=bool)
    filled = np.zeros(line.shape, dtype=bool)
    if int(valid.sum()) == 1:
        line[missing] = line[valid][0]
    else:
        order = np.argsort(coordinate[valid])
        line[missing] = np.interp(coordinate[missing], coordinate[valid][order], line[valid][order])
    filled[missing] = True
    return filled


def repair_missing(values: np.ndarray, times_ms: np.ndarray, siglay: np.ndarray, boundary: Boundary, variable_name: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    work = np.asarray(values, dtype=np.float64).copy()
    if work.shape != (len(times_ms), siglay.shape[0], len(boundary.node_ids)):
        raise ValueError(f"{variable_name} does not match (time, siglay, nobc)")
    original = work.copy()
    original_missing = ~np.isfinite(original)
    codes = np.zeros(work.shape, dtype=np.uint8)
    method_counts = {"temporal": 0, "vertical": 0, "along_boundary": 0}
    passes: list[dict[str, int]] = []
    time_coordinate = times_ms.astype(np.float64)
    max_passes = max(3, sum(work.shape))
    for pass_index in range(max_passes):
        pass_counts = {"temporal": 0, "vertical": 0, "along_boundary": 0}
        for layer in range(work.shape[1]):
            for node in range(work.shape[2]):
                filled = _fill_line(work[:, layer, node], time_coordinate)
                newly = filled & (codes[:, layer, node] == 0)
                codes[:, layer, node][newly] = 1
                pass_counts["temporal"] += int(newly.sum())
        for time_index in range(work.shape[0]):
            for node in range(work.shape[2]):
                filled = _fill_line(work[time_index, :, node], siglay[:, node])
                newly = filled & (codes[time_index, :, node] == 0)
                codes[time_index, :, node][newly] = 2
                pass_counts["vertical"] += int(newly.sum())
        for time_index in range(work.shape[0]):
            for layer in range(work.shape[1]):
                for arc in boundary.arcs:
                    line = work[time_index, layer, arc]
                    filled = _fill_line(line, arc_distances_km(boundary, arc))
                    if np.any(filled):
                        work[time_index, layer, arc] = line
                        local_codes = codes[time_index, layer, arc]
                        newly = filled & (local_codes == 0)
                        local_codes[newly] = 3
                        codes[time_index, layer, arc] = local_codes
                        pass_counts["along_boundary"] += int(newly.sum())
        passes.append({"pass": pass_index + 1, **pass_counts})
        for method in method_counts:
            method_counts[method] += pass_counts[method]
        if sum(pass_counts.values()) == 0:
            break
    unresolved = int((~np.isfinite(work)).sum())
    if unresolved:
        raise ValueError(f"{variable_name} has {unresolved} NaNs that cannot be repaired without inventing data")
    original_finite = np.isfinite(original)
    if not np.array_equal(work[original_finite], original[original_finite]):
        raise AssertionError(f"{variable_name} repair changed original finite values")
    temporal_runs = [_max_consecutive(original_missing[:, layer, node]) for layer in range(work.shape[1]) for node in range(work.shape[2])]
    vertical_runs = [_max_consecutive(original_missing[time_index, :, node]) for time_index in range(work.shape[0]) for node in range(work.shape[2])]
    spatial_runs = [_max_consecutive(original_missing[time_index, layer, arc]) for time_index in range(work.shape[0]) for layer in range(work.shape[1]) for arc in boundary.arcs]
    cadence_ms = int(np.median(np.diff(times_ms)))
    repaired_total = int(original_missing.sum())
    report = {
        "variable": variable_name,
        "original_missing": repaired_total,
        "final_missing": 0,
        "repaired_total": repaired_total,
        "repair_fraction_of_values": repaired_total / int(work.size),
        "method_counts": method_counts,
        "passes": passes,
        "max_original_missing_run": {
            "time_records": int(max(temporal_runs, default=0)),
            "time_hours_at_median_cadence": float(max(temporal_runs, default=0) * cadence_ms / 3_600_000.0),
            "sigma_layers": int(max(vertical_runs, default=0)),
            "same_arc_nodes": int(max(spatial_runs, default=0)),
        },
        "original_finite_values_unchanged": True,
    }
    return work, codes, report


def temporal_resample(source_times_ms: np.ndarray, values: np.ndarray, target_times_ms: np.ndarray, *, max_gap_factor: float = 3.0) -> tuple[np.ndarray, dict[str, Any]]:
    if max_gap_factor <= 1.0:
        raise ValueError("--max-gap-factor must exceed 1")
    if target_times_ms[0] < source_times_ms[0] or target_times_ms[-1] > source_times_ms[-1]:
        raise ValueError("Target period extends outside source coverage; extrapolation is disabled")
    differences = np.diff(source_times_ms)
    native_ms = int(np.median(differences))
    large = differences > max_gap_factor * native_ms
    if np.any(large):
        first = int(np.where(large)[0][0])
        raise ValueError(f"Source time gap {differences[first] / 3_600_000:.3f} h exceeds {max_gap_factor:g} times native cadence")
    if not np.all(np.isfinite(values)):
        raise ValueError("Repair missing values before temporal resampling")
    if np.array_equal(source_times_ms, target_times_ms):
        return values.copy(), {"method": "preserved_source_axis", "source_cadence_seconds": native_ms / 1000.0, "target_cadence_seconds": native_ms / 1000.0, "max_gap_factor": float(max_gap_factor)}
    flat = values.reshape(len(source_times_ms), -1)
    output = np.empty((len(target_times_ms), flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(target_times_ms, source_times_ms, flat[:, column])
    target_cadence = int(np.median(np.diff(target_times_ms)))
    return output.reshape((len(target_times_ms),) + values.shape[1:]), {"method": "linear_no_extrapolation", "source_cadence_seconds": native_ms / 1000.0, "target_cadence_seconds": target_cadence / 1000.0, "max_gap_factor": float(max_gap_factor)}


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


def iso_utc(times_ms: np.ndarray) -> list[str]:
    return [f"{str(np.datetime64(int(value), 'ms'))}Z" for value in np.asarray(times_ms).ravel()]


def write_fvcom_ts_forcing(path: str | Path, boundary: Boundary, times_ms: np.ndarray, siglay: np.ndarray, siglev: np.ndarray, temperature_c: np.ndarray, salinity: np.ndarray, *, case_name: str, source_name: str, source_sha256: str, sigma_orientation: str) -> None:
    output = Path(path)
    shape = (len(times_ms), siglay.shape[0], len(boundary.node_ids))
    if temperature_c.shape != shape or salinity.shape != shape:
        raise ValueError(f"T/S arrays must have shape {shape}")
    if not np.all(np.isfinite(temperature_c)) or not np.all(np.isfinite(salinity)):
        raise ValueError("T/S forcing contains non-finite values")
    if np.any(np.diff(times_ms) <= 0):
        raise ValueError("Output times must be strictly increasing")
    validate_sigma(siglay, siglev)
    delta_ms = times_ms.astype(np.int64) - MJD_EPOCH_MS
    itime = np.floor_divide(delta_ms, DAY_MS).astype(np.int32)
    itime2 = np.mod(delta_ms, DAY_MS).astype(np.int32)
    time_mjd = (itime.astype(np.float64) + itime2.astype(np.float64) / DAY_MS).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    try:
        with nc4.Dataset(temporary, "w", format="NETCDF3_CLASSIC") as dataset:
            dataset.type = "FVCOM TIME SERIES OBC TS FILE"
            dataset.title = str(case_name)
            dataset.history = "Created by fvcom-boundary-ts-forcing"
            dataset.source_file = Path(source_name).name
            dataset.source_sha256 = source_sha256
            dataset.sigma_orientation = sigma_orientation
            dataset.createDimension("nobc", len(boundary.node_ids))
            dataset.createDimension("siglay", siglay.shape[0])
            dataset.createDimension("siglev", siglev.shape[0])
            dataset.createDimension("time", None)
            dataset.createDimension("DateStrLen", 26)
            variable = dataset.createVariable("obc_nodes", "i4", ("nobc",))
            variable.long_name = "Open Boundary Node Number"
            variable.grid = "obc_grid"
            variable[:] = boundary.node_ids.astype(np.int32)
            variable = dataset.createVariable("obc_h", "f4", ("nobc",))
            variable.long_name = "Bathymetry at Open Boundary Nodes"
            variable.units = "meters"
            variable.positive = "down"
            variable[:] = boundary.depth_m.astype(np.float32)
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
            variable[:] = format_times(times_ms)
            variable = dataset.createVariable("siglay", "f4", ("siglay", "nobc"))
            variable.long_name = "Sigma Layers"
            variable.units = "sigma_layers"
            variable[:] = siglay.astype(np.float32)
            variable = dataset.createVariable("siglev", "f4", ("siglev", "nobc"))
            variable.long_name = "Sigma Levels"
            variable.units = "sigma_levels"
            variable[:] = siglev.astype(np.float32)
            variable = dataset.createVariable("obc_temp", "f4", ("time", "siglay", "nobc"))
            variable.long_name = "Open Boundary Temperature"
            variable.units = "Celsius"
            variable[:] = temperature_c.astype(np.float32)
            variable = dataset.createVariable("obc_salinity", "f4", ("time", "siglay", "nobc"))
            variable.long_name = "Open Boundary Salinity"
            variable.units = "PSU"
            variable[:] = salinity.astype(np.float32)
        os.replace(temporary, output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
