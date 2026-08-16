from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import netCDF4 as nc4
import numpy as np


MJD_EPOCH_MS = int(np.datetime64("1858-11-17T00:00:00", "ms").astype(np.int64))
DAY_MS = 86_400_000
PACKAGES = ("wind", "heat", "freshwater", "pressure")
SUFFIXES = {"wind": "wnd", "heat": "hfx", "freshwater": "emp", "pressure": "aip"}

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "eastward_wind": ("eastward_wind", "U10", "u10", "uwind_speed", "wndewd"),
    "northward_wind": ("northward_wind", "V10", "v10", "vwind_speed", "wndnwd"),
    "eastward_stress": ("eastward_stress", "uwind_stress", "Stress_U", "tauewd"),
    "northward_stress": ("northward_stress", "vwind_stress", "Stress_V", "taunwd"),
    "net_shortwave": ("net_shortwave", "net_short_wave", "shwflx"),
    "total_net_heat_flux": ("total_net_heat_flux", "net_heat_flux"),
    "air_temperature": ("air_temperature", "air_temp", "airtmp", "T2"),
    "relative_humidity": ("relative_humidity", "rh", "RH2"),
    "absolute_air_pressure": ("absolute_air_pressure", "air_pressure", "pressure_air", "SLP"),
    "downward_longwave": ("downward_longwave", "long_wave", "Longwave", "dlwflx"),
    "downward_shortwave": ("downward_shortwave", "short_wave", "Shortwave", "dswflx"),
    "precipitation": ("precipitation", "Precipitation", "precip"),
    "evaporation": ("evaporation", "Evaporation", "evap"),
}
TIME_ALIASES = ("time", "MT", "Times", "datetime", "timestamp")
LAT_ALIASES = ("latitude", "lat", "XLAT")
LON_ALIASES = ("longitude", "lon", "XLONG")
NODE_ID_ALIASES = ("node_id", "node_ids", "node")
ELEMENT_ID_ALIASES = ("element_id", "element_ids", "nele")


@dataclass(frozen=True)
class MeshGeometry:
    node_ids: np.ndarray
    element_ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    lonc: np.ndarray
    latc: np.ndarray
    triangles: np.ndarray
    source: str


@dataclass(frozen=True)
class PreparedSurfaceData:
    layout: str
    times_ms: np.ndarray
    fields: dict[str, np.ndarray]
    field_units: dict[str, str]
    lat: np.ndarray
    lon: np.ndarray
    mesh: MeshGeometry | None
    source: str
    source_sha256: str
    transformations: tuple[str, ...]


@dataclass(frozen=True)
class BundleResult:
    files: dict[str, Path]
    package_files: dict[str, Path]
    namelist: str
    transformations: tuple[str, ...]
    layout_decision: str


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
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


def _first_present(candidates: Iterable[str], available: Iterable[str]) -> str | None:
    lookup = {str(value).lower(): str(value) for value in available}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _validate_geographic(lon: np.ndarray, lat: np.ndarray, label: str) -> None:
    if lon.shape != lat.shape or lon.size == 0:
        raise ValueError(f"{label} longitude/latitude shapes must match and be non-empty")
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError(f"{label} coordinates contain non-finite values")
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 360.0):
        raise ValueError(f"{label} coordinates must be geographic longitude/latitude degrees")


def read_mesh_2dm(path: str | Path) -> MeshGeometry:
    mesh_path = Path(path)
    nodes: list[tuple[int, float, float]] = []
    elements: list[tuple[int, int, int, int]] = []
    with mesh_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            parts = raw.split()
            if not parts:
                continue
            record = parts[0].upper()
            if record == "ND":
                if len(parts) < 5:
                    raise ValueError(f"Malformed ND record at {mesh_path}:{line_number}")
                nodes.append((int(parts[1]), float(parts[2]), float(parts[3])))
            elif record == "E3T":
                if len(parts) < 5:
                    raise ValueError(f"Malformed E3T record at {mesh_path}:{line_number}")
                elements.append(tuple(int(value) for value in parts[1:5]))
    return _mesh_from_records(nodes, elements, str(mesh_path))


def read_mesh_grd(path: str | Path) -> MeshGeometry:
    grid_path = Path(path)
    lines = grid_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Incomplete FVCOM grid file: {grid_path}")
    try:
        node_count = int(lines[0].split("=")[-1])
        element_count = int(lines[1].split("=")[-1])
    except Exception as exc:
        raise ValueError(f"Cannot parse node/element counts in {grid_path}") from exc
    element_lines = lines[2 : 2 + element_count]
    node_lines = lines[2 + element_count : 2 + element_count + node_count]
    if len(element_lines) != element_count or len(node_lines) != node_count:
        raise ValueError(f"Grid record counts do not match headers in {grid_path}")
    elements = [tuple(int(value) for value in raw.split()[:4]) for raw in element_lines]
    nodes = [(int(p[0]), float(p[1]), float(p[2])) for p in (raw.split() for raw in node_lines)]
    return _mesh_from_records(nodes, elements, str(grid_path))


def _mesh_from_records(
    nodes: Sequence[tuple[int, float, float]],
    elements: Sequence[tuple[int, int, int, int]],
    source: str,
) -> MeshGeometry:
    if not nodes or not elements:
        raise ValueError(f"Mesh {source} must contain nodes and triangular elements")
    node_ids = np.asarray([row[0] for row in nodes], dtype=np.int64)
    element_ids = np.asarray([row[0] for row in elements], dtype=np.int64)
    if len(np.unique(node_ids)) != len(node_ids) or len(np.unique(element_ids)) != len(element_ids):
        raise ValueError(f"Mesh {source} contains duplicate node or element ids")
    lon = np.asarray([row[1] for row in nodes], dtype=np.float64)
    lat = np.asarray([row[2] for row in nodes], dtype=np.float64)
    _validate_geographic(lon, lat, "Mesh")
    positions = {int(node_id): index for index, node_id in enumerate(node_ids)}
    triangles = np.empty((len(elements), 3), dtype=np.int64)
    for index, row in enumerate(elements):
        try:
            triangles[index] = [positions[row[1]], positions[row[2]], positions[row[3]]]
        except KeyError as exc:
            raise ValueError(f"Element {row[0]} references missing node {exc.args[0]}") from exc
    lonc = np.mean(lon[triangles], axis=1)
    latc = np.mean(lat[triangles], axis=1)
    return MeshGeometry(node_ids, element_ids, lon, lat, lonc, latc, triangles, source)


def _parse_time_text(text: str, assume_utc: bool) -> int:
    value = text.strip().replace("/", "-")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        if not assume_utc:
            raise ValueError(f"Timestamp {text!r} is timezone-free; pass --assume-utc only after confirming UTC")
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(round(parsed.astimezone(timezone.utc).timestamp() * 1000.0))


def _decode_times(dataset: nc4.Dataset, override: str | None, assume_utc: bool) -> tuple[np.ndarray, str]:
    variables = dataset.variables
    name = override or _first_present(TIME_ALIASES, variables)
    if name == "Times" or (name is None and "Times" in variables):
        name = "Times"
        texts = nc4.chartostring(variables[name][:])
        values = np.asarray([_parse_time_text(str(value), True) for value in texts], dtype=np.int64)
    elif name in {"Itime", "Itime2"} or (name is None and "Itime" in variables and "Itime2" in variables):
        days = np.asarray(variables["Itime"][:], dtype=np.int64)
        millis = np.asarray(variables["Itime2"][:], dtype=np.int64)
        values = MJD_EPOCH_MS + days * DAY_MS + millis
        name = "Itime+Itime2"
    elif name and name in variables:
        variable = variables[name]
        if variable.dtype.kind in {"S", "U"}:
            texts = nc4.chartostring(variable[:]) if variable.ndim == 2 else variable[:]
            values = np.asarray([_parse_time_text(str(value), assume_utc) for value in texts], dtype=np.int64)
        else:
            units = getattr(variable, "units", None)
            if not units:
                raise ValueError(f"Numeric time variable {name!r} has no units")
            calendar = getattr(variable, "calendar", "standard")
            dates = nc4.num2date(variable[:], units=units, calendar=calendar)
            values = np.asarray(
                [
                    int(
                        round(
                            datetime(
                                item.year,
                                item.month,
                                item.day,
                                item.hour,
                                item.minute,
                                item.second,
                                item.microsecond,
                                tzinfo=timezone.utc,
                            ).timestamp()
                            * 1000.0
                        )
                    )
                    for item in dates
                ],
                dtype=np.int64,
            )
    else:
        raise ValueError("Could not identify a UTC-decodable time variable")
    if values.ndim != 1 or not len(values) or np.any(np.diff(values) <= 0):
        raise ValueError("Source time must be one-dimensional and strictly increasing")
    return values, str(name)


def _masked_float(variable: Any) -> np.ndarray:
    value = variable[:]
    if np.ma.isMaskedArray(value):
        value = value.filled(np.nan)
    output = np.asarray(value, dtype=np.float64)
    fill = getattr(variable, "_FillValue", None)
    if fill is not None:
        output[output == float(fill)] = np.nan
    missing = getattr(variable, "missing_value", None)
    if missing is not None:
        output[output == float(np.asarray(missing).ravel()[0])] = np.nan
    return output


def _unit_key(units: str) -> str:
    return units.strip().lower().replace(" ", "").replace("**", "^")


def convert_field(role: str, values: np.ndarray, units: str, pressure_reference: str | None) -> tuple[np.ndarray, str, str | None]:
    key = _unit_key(units)
    data = np.asarray(values, dtype=np.float64)
    note: str | None = None
    if role in {"eastward_wind", "northward_wind"}:
        if key not in {"m/s", "ms-1", "m.s-1", "metersecond-1", "meterssecond-1"}:
            raise ValueError(f"{role} requires m/s, found {units!r}")
        canonical = "m s-1"
    elif role in {"eastward_stress", "northward_stress"}:
        if key not in {"pa", "n/m2", "nm-2", "n.m-2"}:
            raise ValueError(f"{role} requires Pa or N/m2, found {units!r}")
        canonical = "Pa"
    elif role in {"net_shortwave", "total_net_heat_flux", "downward_longwave", "downward_shortwave"}:
        if key not in {"w/m2", "wm-2", "w.m-2"}:
            raise ValueError(f"{role} requires W/m2, found {units!r}")
        canonical = "W m-2"
    elif role == "air_temperature":
        if key in {"k", "kelvin", "degrees_k", "degree_k"}:
            data = data - 273.15
            note = "air_temperature: Kelvin to Celsius"
        elif key not in {"c", "degc", "degree_celsius", "degreescelsius", "celsius"}:
            raise ValueError(f"air_temperature requires Kelvin or Celsius, found {units!r}")
        canonical = "Celsius"
    elif role == "relative_humidity":
        if key in {"1", "fraction", "unitless", ""}:
            data = data * 100.0
            note = "relative_humidity: fraction to percent"
        elif key not in {"%", "percent", "percentage"}:
            raise ValueError(f"relative_humidity requires percent or fraction, found {units!r}")
        canonical = "percent"
    elif role == "absolute_air_pressure":
        if str(pressure_reference or "").strip().lower() != "absolute":
            raise ValueError("Atmospheric pressure must declare pressure_reference='absolute'; departures are not accepted")
        if key in {"hpa", "mb", "mbar", "hectopascal", "hectopascals"}:
            data = data * 100.0
            note = "absolute_air_pressure: hPa to Pa"
        elif key not in {"pa", "pascal", "pascals"}:
            raise ValueError(f"absolute_air_pressure requires Pa or hPa, found {units!r}")
        canonical = "Pa"
    elif role in {"precipitation", "evaporation"}:
        if key in {"m/s", "ms-1", "m.s-1"}:
            pass
        elif key in {"kgm-2s-1", "kg/m2/s", "kgm^-2s^-1"}:
            data = data / 1000.0
            note = f"{role}: kg m-2 s-1 to m s-1 using 1000 kg m-3 freshwater density"
        elif key in {"mm/hr", "mmh-1", "mm/hour"}:
            data = data / 1000.0 / 3600.0
            note = f"{role}: mm hr-1 to m s-1"
        elif key in {"mm/day", "mmd-1", "mmday-1"}:
            data = data / 1000.0 / 86400.0
            note = f"{role}: mm day-1 to m s-1"
        else:
            raise ValueError(f"{role} requires water-depth or mass flux units, found {units!r}")
        canonical = "m s-1"
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"Unknown field role {role}")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{role} contains NaN, infinity, or fill values")
    if role == "relative_humidity" and (np.min(data) < 0.0 or np.max(data) > 100.0):
        raise ValueError("relative_humidity must remain within 0-100 percent")
    if role == "absolute_air_pressure" and (np.min(data) < 50_000.0 or np.max(data) > 120_000.0):
        raise ValueError("absolute_air_pressure is outside the 50,000-120,000 Pa safety range")
    if role == "air_temperature" and (np.min(data) < -100.0 or np.max(data) > 70.0):
        raise ValueError("air_temperature is outside the -100 to 70 Celsius safety range")
    if role in {"precipitation", "evaporation"} and np.min(data) < 0.0:
        raise ValueError(f"Prepared {role} must be a non-negative magnitude")
    return data, canonical, note


def required_roles(packages: Sequence[str], wind_mode: str, heat_mode: str) -> tuple[str, ...]:
    roles: list[str] = []
    if "wind" in packages:
        roles.extend(
            ("eastward_wind", "northward_wind")
            if wind_mode == "speed"
            else ("eastward_stress", "northward_stress")
        )
    if "heat" in packages:
        roles.extend(
            ("net_shortwave", "total_net_heat_flux")
            if heat_mode == "direct"
            else (
                "air_temperature",
                "relative_humidity",
                "absolute_air_pressure",
                "downward_longwave",
                "downward_shortwave",
            )
        )
    if "freshwater" in packages:
        roles.extend(("precipitation", "evaporation"))
    if "pressure" in packages and "absolute_air_pressure" not in roles:
        roles.append("absolute_air_pressure")
    return tuple(roles)


def _validate_modes(packages: Sequence[str], wind_mode: str, heat_mode: str, coare_version: str) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(value).lower() for value in packages))
    if not selected or any(value not in PACKAGES for value in selected):
        raise ValueError(f"Packages must be a non-empty subset of {PACKAGES}")
    if wind_mode not in {"speed", "stress"}:
        raise ValueError("wind_mode must be speed or stress")
    if heat_mode not in {"direct", "bulk"}:
        raise ValueError("heat_mode must be direct or bulk")
    if coare_version not in {"COARE26Z", "COARE40VN"}:
        raise ValueError("coare_version must be COARE26Z or COARE40VN")
    return selected


def _structured_coordinates(dataset: nc4.Dataset, lat_name: str | None, lon_name: str | None) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    lat_key = lat_name or _first_present(LAT_ALIASES, dataset.variables)
    lon_key = lon_name or _first_present(LON_ALIASES, dataset.variables)
    if not lat_key or not lon_key:
        raise ValueError("Structured input requires latitude and longitude coordinates")
    lat_var, lon_var = dataset.variables[lat_key], dataset.variables[lon_key]
    lat = _masked_float(lat_var)
    lon = _masked_float(lon_var)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        dims = (lat_var.dimensions[0], lon_var.dimensions[0])
    elif lat.ndim == 2 and lon.ndim == 2 and lat.shape == lon.shape and lat_var.dimensions == lon_var.dimensions:
        lat2d, lon2d = lat, lon
        dims = tuple(lat_var.dimensions)
    else:
        raise ValueError("Structured latitude/longitude must be matching 2-D arrays or separate 1-D axes")
    _validate_geographic(lon2d, lat2d, "Structured grid")
    return lat2d, lon2d, (str(dims[0]), str(dims[1]))


def _read_role_variable(
    dataset: nc4.Dataset,
    role: str,
    override: Mapping[str, str],
    time_dim: str,
    spatial_dims: tuple[str, ...],
) -> tuple[np.ndarray, str, str | None]:
    name = override.get(role) or _first_present(ROLE_ALIASES[role], dataset.variables)
    if not name or name not in dataset.variables:
        raise ValueError(f"Missing prepared field for role {role!r}; use a canonical name or --var {role}=NAME")
    variable = dataset.variables[name]
    expected = (time_dim, *spatial_dims)
    if set(variable.dimensions) != set(expected) or len(variable.dimensions) != len(expected):
        raise ValueError(f"{name} dimensions {variable.dimensions} do not match expected {expected}")
    order = [variable.dimensions.index(dim) for dim in expected]
    values = np.transpose(_masked_float(variable), axes=order)
    units = getattr(variable, "units", "")
    pressure_reference = getattr(variable, "pressure_reference", None)
    return values, str(units), pressure_reference


def read_prepared_netcdf(
    path: str | Path,
    *,
    layout: str,
    packages: Sequence[str],
    wind_mode: str = "speed",
    heat_mode: str = "direct",
    coare_version: str = "COARE26Z",
    mesh: MeshGeometry | None = None,
    var_map: Mapping[str, str] | None = None,
    time_var: str | None = None,
    lat_var: str | None = None,
    lon_var: str | None = None,
    pressure_reference: str | None = None,
    assume_utc: bool = False,
) -> PreparedSurfaceData:
    selected = _validate_modes(packages, wind_mode, heat_mode, coare_version)
    if layout not in {"structured", "fvcom"}:
        raise ValueError("layout must be structured or fvcom")
    source_path = Path(path)
    transformations: list[str] = []
    override = dict(var_map or {})
    with nc4.Dataset(source_path) as dataset:
        times_ms, time_source = _decode_times(dataset, time_var, assume_utc)
        time_dim = dataset.variables[time_var].dimensions[0] if time_var else None
        if time_dim is None:
            candidate = _first_present(TIME_ALIASES, dataset.variables)
            if candidate and dataset.variables[candidate].dimensions:
                time_dim = dataset.variables[candidate].dimensions[0]
            elif "Itime" in dataset.variables:
                time_dim = dataset.variables["Itime"].dimensions[0]
        if not time_dim:
            raise ValueError("Could not identify the source time dimension")

        if layout == "structured":
            lat, lon, grid_dims = _structured_coordinates(dataset, lat_var, lon_var)
            role_dims = {role: grid_dims for role in required_roles(selected, wind_mode, heat_mode)}
            mesh_value = None
        else:
            if mesh is None:
                raise ValueError("FVCOM-native input requires a parsed mesh")
            node_key = _first_present(NODE_ID_ALIASES, dataset.variables)
            element_key = _first_present(ELEMENT_ID_ALIASES, dataset.variables)
            if not node_key or not element_key:
                raise ValueError("FVCOM-native input requires node_id and element_id coordinates")
            source_nodes = np.asarray(dataset.variables[node_key][:], dtype=np.int64)
            source_elements = np.asarray(dataset.variables[element_key][:], dtype=np.int64)
            if not np.array_equal(source_nodes, mesh.node_ids):
                raise ValueError("Prepared node_id order does not exactly match the mesh")
            if not np.array_equal(source_elements, mesh.element_ids):
                raise ValueError("Prepared element_id order does not exactly match the mesh")
            node_dim = dataset.variables[node_key].dimensions[0]
            element_dim = dataset.variables[element_key].dimensions[0]
            role_dims = {
                role: ((element_dim,) if role in {"eastward_wind", "northward_wind", "eastward_stress", "northward_stress"} else (node_dim,))
                for role in required_roles(selected, wind_mode, heat_mode)
            }
            lat, lon, mesh_value = mesh.lat, mesh.lon, mesh

        fields: dict[str, np.ndarray] = {}
        units_out: dict[str, str] = {}
        global_pressure_reference = pressure_reference or getattr(dataset, "pressure_reference", None)
        for role in required_roles(selected, wind_mode, heat_mode):
            values, units, variable_reference = _read_role_variable(
                dataset, role, override, str(time_dim), role_dims[role]
            )
            converted, canonical, note = convert_field(
                role, values, units, pressure_reference or variable_reference or global_pressure_reference
            )
            fields[role] = converted
            units_out[role] = canonical
            if note:
                transformations.append(note)
        transformations.insert(0, f"time decoded from {time_source} as UTC")
    return PreparedSurfaceData(
        layout,
        times_ms,
        fields,
        units_out,
        lat,
        lon,
        mesh_value,
        str(source_path),
        sha256_file(source_path),
        tuple(transformations),
    )


def _coerce_times(values: Sequence[Any]) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        result = array.astype("datetime64[ms]").astype(np.int64)
    else:
        result = np.asarray([_parse_time_text(str(value), False) for value in values], dtype=np.int64)
    if result.ndim != 1 or not len(result) or np.any(np.diff(result) <= 0):
        raise ValueError("times_utc must be a strictly increasing one-dimensional UTC axis")
    return result


def prepare_arrays(
    *,
    layout: str,
    times_utc: Sequence[Any],
    fields: Mapping[str, np.ndarray],
    units: Mapping[str, str],
    packages: Sequence[str],
    wind_mode: str = "speed",
    heat_mode: str = "direct",
    coare_version: str = "COARE26Z",
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    mesh: MeshGeometry | None = None,
    pressure_reference: str | None = None,
    source: str = "python_api",
) -> PreparedSurfaceData:
    selected = _validate_modes(packages, wind_mode, heat_mode, coare_version)
    times_ms = _coerce_times(times_utc)
    if layout == "structured":
        if lat is None or lon is None:
            raise ValueError("Structured API input requires latitude and longitude")
        lat_value, lon_value = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
        if lat_value.ndim == 1 and lon_value.ndim == 1:
            lon_value, lat_value = np.meshgrid(lon_value, lat_value)
        _validate_geographic(lon_value, lat_value, "Structured API grid")
        expected = (len(times_ms), *lat_value.shape)
    elif layout == "fvcom":
        if mesh is None:
            raise ValueError("FVCOM-native API input requires MeshGeometry")
        lat_value, lon_value = mesh.lat, mesh.lon
        expected = None
    else:
        raise ValueError("layout must be structured or fvcom")
    converted_fields: dict[str, np.ndarray] = {}
    converted_units: dict[str, str] = {}
    transformations: list[str] = ["time supplied through Python API as UTC"]
    for role in required_roles(selected, wind_mode, heat_mode):
        if role not in fields or role not in units:
            raise ValueError(f"Python API input is missing field or units for {role}")
        raw = np.asarray(fields[role])
        role_expected = expected
        if layout == "fvcom":
            width = len(mesh.element_ids) if role in {"eastward_wind", "northward_wind", "eastward_stress", "northward_stress"} else len(mesh.node_ids)
            role_expected = (len(times_ms), width)
        if raw.shape != role_expected:
            raise ValueError(f"{role} shape {raw.shape} does not match expected {role_expected}")
        converted, canonical, note = convert_field(role, raw, units[role], pressure_reference)
        converted_fields[role], converted_units[role] = converted, canonical
        if note:
            transformations.append(note)
    return PreparedSurfaceData(
        layout,
        times_ms,
        converted_fields,
        converted_units,
        lat_value,
        lon_value,
        mesh,
        source,
        "",
        tuple(transformations),
    )


def format_times(times_ms: np.ndarray) -> np.ndarray:
    output = np.zeros((len(times_ms), 26), dtype="S1")
    for index, value in enumerate(times_ms):
        stamp = str(np.datetime64(int(value), "ms")).replace("T", " ")
        if "." not in stamp:
            stamp += ".000"
        date, fraction = stamp.split(".", 1)
        text = f"{date.replace('-', '/')}.{fraction[:3].ljust(3, '0')}000"
        output[index] = np.asarray(list(text[:26]), dtype="S1")
    return output


def _output_groups(
    packages: tuple[str, ...], file_layout: str, heat_mode: str, coare_version: str
) -> tuple[list[tuple[str, tuple[str, ...]]], str]:
    if file_layout not in {"auto", "combined", "split"}:
        raise ValueError("file_layout must be auto, combined, or split")
    unsafe = heat_mode == "bulk" and coare_version == "COARE40VN" and {"heat", "pressure"}.issubset(packages)
    if file_layout == "combined" and unsafe:
        raise ValueError("COARE40VN heat needs hPa while inverse-barometer pressure needs Pa; a combined file is unsafe")
    split = file_layout == "split" or (file_layout == "auto" and unsafe)
    if split:
        reason = "split_requested" if file_layout == "split" else "split_required_for_COARE40VN_pressure_units"
        return [(package, (package,)) for package in packages], reason
    if len(packages) == 1:
        return [(packages[0], packages)], "single_package"
    return [("surface", packages)], "combined_compatible_packages"


def _time_variables(dataset: nc4.Dataset, times_ms: np.ndarray) -> None:
    delta = times_ms - MJD_EPOCH_MS
    itime = np.floor_divide(delta, DAY_MS).astype(np.int32)
    itime2 = np.mod(delta, DAY_MS).astype(np.int32)
    mjd = (itime.astype(np.float64) + itime2.astype(np.float64) / DAY_MS).astype(np.float32)
    variable = dataset.createVariable("iint", "i4", ("time",))
    variable.long_name = "internal mode iteration number"
    variable[:] = np.arange(1, len(times_ms) + 1, dtype=np.int32)
    variable = dataset.createVariable("time", "f4", ("time",))
    variable.units = "days since 1858-11-17 00:00:00"
    variable.format = "modified julian day (MJD)"
    variable.time_zone = "UTC"
    variable[:] = mjd
    variable = dataset.createVariable("Itime", "i4", ("time",))
    variable.units = "days since 1858-11-17 00:00:00"
    variable.time_zone = "UTC"
    variable[:] = itime
    variable = dataset.createVariable("Itime2", "i4", ("time",))
    variable.units = "msec since 00:00:00"
    variable.time_zone = "UTC"
    variable[:] = itime2
    variable = dataset.createVariable("Times", "S1", ("time", "DateStrLen"))
    variable.time_zone = "UTC"
    variable[:] = format_times(times_ms)


def _package_variables(
    packages: Sequence[str], layout: str, wind_mode: str, heat_mode: str, coare_version: str
) -> list[tuple[str, str, str, str, float]]:
    result: list[tuple[str, str, str, str, float]] = []
    if "wind" in packages:
        if wind_mode == "speed":
            names = ("U10", "V10") if layout == "structured" else ("uwind_speed", "vwind_speed")
            result.extend((("eastward_wind", names[0], "m s-1", "element", 1.0), ("northward_wind", names[1], "m s-1", "element", 1.0)))
        else:
            result.extend((("eastward_stress", "uwind_stress", "Pa", "element", 1.0), ("northward_stress", "vwind_stress", "Pa", "element", 1.0)))
    if "heat" in packages:
        if heat_mode == "direct":
            result.extend((("net_shortwave", "short_wave", "W m-2", "node", 1.0), ("total_net_heat_flux", "net_heat_flux", "W m-2", "node", 1.0)))
        else:
            pressure_scale = 0.01 if coare_version == "COARE40VN" else 1.0
            pressure_units = "hPa" if coare_version == "COARE40VN" else "Pa"
            result.extend(
                (
                    ("air_temperature", "air_temperature", "Celsius", "node", 1.0),
                    ("relative_humidity", "relative_humidity", "percent", "node", 1.0),
                    ("absolute_air_pressure", "air_pressure", pressure_units, "node", pressure_scale),
                    ("downward_longwave", "long_wave", "W m-2", "node", 1.0),
                    ("downward_shortwave", "short_wave", "W m-2", "node", 1.0),
                )
            )
    if "freshwater" in packages:
        names = ("Precipitation", "Evaporation") if layout == "structured" else ("precip", "evap")
        result.extend((("precipitation", names[0], "m s-1", "node", 1.0), ("evaporation", names[1], "m s-1", "node", -1.0)))
    if "pressure" in packages:
        result.append(("absolute_air_pressure", "air_pressure", "Pa", "node", 1.0))
    unique: dict[str, tuple[str, str, str, str, float]] = {}
    for item in result:
        if item[1] in unique and unique[item[1]][2:] != item[2:]:
            raise ValueError(f"Output variable {item[1]} has incompatible package contracts")
        unique[item[1]] = item
    return list(unique.values())


def _write_one(
    path: Path,
    data: PreparedSurfaceData,
    packages: Sequence[str],
    case_name: str,
    wind_mode: str,
    heat_mode: str,
    coare_version: str,
) -> None:
    with nc4.Dataset(path, "w", format="NETCDF3_CLASSIC") as dataset:
        dataset.type = "FVCOM SURFACE FORCING FILE"
        dataset.title = case_name
        dataset.history = "Created by fvcom-surface-fluxes-forcing"
        dataset.source = "wrf grid (structured) surface forcing" if data.layout == "structured" else "FVCOM grid (unstructured) surface forcing"
        dataset.source_file = Path(data.source).name
        dataset.source_sha256 = data.source_sha256
        dataset.active_packages = ",".join(packages)
        dataset.wind_mode = wind_mode
        dataset.heat_mode = heat_mode
        dataset.coare_version = coare_version
        dataset.createDimension("time", None)
        dataset.createDimension("DateStrLen", 26)
        if data.layout == "structured":
            ny, nx = data.lat.shape
            dataset.createDimension("south_north", ny)
            dataset.createDimension("west_east", nx)
            variable = dataset.createVariable("XLAT", "f4", ("south_north", "west_east"))
            variable.units = "degrees_north"
            variable[:] = data.lat.astype(np.float32)
            variable = dataset.createVariable("XLONG", "f4", ("south_north", "west_east"))
            variable.units = "degrees_east"
            variable[:] = data.lon.astype(np.float32)
        else:
            assert data.mesh is not None
            dataset.createDimension("node", len(data.mesh.node_ids))
            dataset.createDimension("nele", len(data.mesh.element_ids))
            for name, values, dim, units in (
                ("node_id", data.mesh.node_ids, "node", "1"),
                ("element_id", data.mesh.element_ids, "nele", "1"),
                ("lon", data.mesh.lon, "node", "degrees_east"),
                ("lat", data.mesh.lat, "node", "degrees_north"),
                ("lonc", data.mesh.lonc, "nele", "degrees_east"),
                ("latc", data.mesh.latc, "nele", "degrees_north"),
            ):
                dtype = "i4" if name.endswith("_id") else "f4"
                variable = dataset.createVariable(name, dtype, (dim,))
                variable.units = units
                variable[:] = values.astype(np.int32 if dtype == "i4" else np.float32)
        _time_variables(dataset, data.times_ms)
        for role, name, units, location, scale in _package_variables(packages, data.layout, wind_mode, heat_mode, coare_version):
            dimensions = ("time", "south_north", "west_east") if data.layout == "structured" else ("time", "nele" if location == "element" else "node")
            variable = dataset.createVariable(name, "f4", dimensions)
            variable.units = units
            variable.grid = "fvcom_grid" if data.layout == "fvcom" else "structured_grid"
            variable.source_role = role
            variable[:] = (data.fields[role] * scale).astype(np.float32)


def _validate_staged(path: Path, data: PreparedSurfaceData, packages: Sequence[str], wind_mode: str, heat_mode: str, coare_version: str) -> None:
    with nc4.Dataset(path) as dataset:
        if dataset.data_model != "NETCDF3_CLASSIC":
            raise ValueError(f"Staged file {path.name} is not NETCDF3_CLASSIC")
        if len(dataset.dimensions["time"]) != len(data.times_ms) or not dataset.dimensions["time"].isunlimited():
            raise ValueError(f"Staged file {path.name} has an invalid time dimension")
        days = np.asarray(dataset.variables["Itime"][:], dtype=np.int64)
        millis = np.asarray(dataset.variables["Itime2"][:], dtype=np.int64)
        if not np.array_equal(MJD_EPOCH_MS + days * DAY_MS + millis, data.times_ms):
            raise ValueError(f"Staged file {path.name} lost exact UTC time")
        for _role, name, _units, _location, _scale in _package_variables(packages, data.layout, wind_mode, heat_mode, coare_version):
            values = _masked_float(dataset.variables[name])
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Staged file {path.name}:{name} contains missing values")


def _namelist(package_files: Mapping[str, Path], packages: Sequence[str], wind_mode: str, heat_mode: str, coare_version: str) -> str:
    def file_for(package: str) -> str:
        return package_files[package].name if package in package_files else "unused.nc"

    direct = "heat" in packages and heat_mode == "direct"
    bulk = "heat" in packages and heat_mode == "bulk"
    lines = [
        "&NML_SURFACE_FORCING",
        f" WIND_ON = {'T' if 'wind' in packages else 'F'},",
        f" WIND_TYPE = '{wind_mode}',",
        f" WIND_FILE = '{file_for('wind')}',",
        " WIND_KIND = 'variable',",
        f" HEATING_ON = {'T' if direct else 'F'},",
        " HEATING_TYPE = 'flux',",
        " HEATING_KIND = 'variable',",
        f" HEATING_FILE = '{file_for('heat')}',",
        f" PRECIPITATION_ON = {'T' if 'freshwater' in packages else 'F'},",
        " PRECIPITATION_KIND = 'variable',",
        f" PRECIPITATION_FILE = '{file_for('freshwater')}',",
        f" AIRPRESSURE_ON = {'T' if 'pressure' in packages else 'F'},",
        " AIRPRESSURE_KIND = 'variable',",
        f" AIRPRESSURE_FILE = '{file_for('pressure')}',",
        "/",
        "",
        "&NML_HEATING_CALCULATED",
        f" HEATING_CALCULATE_ON = {'T' if bulk else 'F'},",
        " HEATING_CALCULATE_TYPE = 'flux',",
        " HEATING_CALCULATE_KIND = 'variable',",
        f" HEATING_CALCULATE_FILE = '{file_for('heat')}',",
        f" COARE_VERSION = '{coare_version}',",
        "/",
    ]
    return "\n".join(lines) + "\n"


def write_prepared_bundle(
    data: PreparedSurfaceData,
    output_dir: str | Path,
    *,
    case_name: str,
    packages: Sequence[str],
    wind_mode: str = "speed",
    heat_mode: str = "direct",
    coare_version: str = "COARE26Z",
    file_layout: str = "auto",
    external_wind_speed: bool = False,
    model_start_ms: int | None = None,
    model_end_ms: int | None = None,
) -> BundleResult:
    selected = _validate_modes(packages, wind_mode, heat_mode, coare_version)
    if heat_mode == "bulk" and "heat" in selected:
        has_speed = "wind" in selected and wind_mode == "speed"
        if not has_speed and not external_wind_speed:
            raise ValueError("Bulk heat requires a selected wind-speed package or external_wind_speed=True")
    if model_start_ms is not None and data.times_ms[0] > model_start_ms:
        raise ValueError("Forcing begins after the requested FVCOM model start")
    if model_end_ms is not None and data.times_ms[-1] < model_end_ms:
        raise ValueError("Forcing ends before the requested FVCOM model end")
    groups, decision = _output_groups(selected, file_layout, heat_mode, coare_version)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path, str, tuple[str, ...]]] = []
    files: dict[str, Path] = {}
    package_files: dict[str, Path] = {}
    try:
        for label, group_packages in groups:
            suffix = "surface" if label == "surface" else SUFFIXES[label]
            destination = directory / f"{case_name}_{suffix}.nc"
            fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=directory)
            os.close(fd)
            temporary = Path(temporary_name)
            staged.append((temporary, destination, label, group_packages))
            _write_one(temporary, data, group_packages, case_name, wind_mode, heat_mode, coare_version)
            _validate_staged(temporary, data, group_packages, wind_mode, heat_mode, coare_version)
        for temporary, destination, label, group_packages in staged:
            os.replace(temporary, destination)
            files[label] = destination
            for package in group_packages:
                package_files[package] = destination
    except Exception:
        for temporary, _destination, _label, _packages in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise
    return BundleResult(
        files,
        package_files,
        _namelist(package_files, selected, wind_mode, heat_mode, coare_version),
        data.transformations,
        decision,
    )


def write_surface_forcing_bundle(
    output_dir: str | Path,
    case_name: str,
    *,
    layout: str,
    times_utc: Sequence[Any],
    fields: Mapping[str, np.ndarray],
    units: Mapping[str, str],
    packages: Sequence[str],
    wind_mode: str = "speed",
    heat_mode: str = "direct",
    coare_version: str = "COARE26Z",
    file_layout: str = "auto",
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    mesh: MeshGeometry | None = None,
    pressure_reference: str | None = None,
    source: str = "python_api",
    external_wind_speed: bool = False,
    model_start_ms: int | None = None,
    model_end_ms: int | None = None,
) -> BundleResult:
    """Public NumPy API for writing a validated modular forcing bundle."""
    prepared = prepare_arrays(
        layout=layout,
        times_utc=times_utc,
        fields=fields,
        units=units,
        packages=packages,
        wind_mode=wind_mode,
        heat_mode=heat_mode,
        coare_version=coare_version,
        lat=lat,
        lon=lon,
        mesh=mesh,
        pressure_reference=pressure_reference,
        source=source,
    )
    return write_prepared_bundle(
        prepared,
        output_dir,
        case_name=case_name,
        packages=packages,
        wind_mode=wind_mode,
        heat_mode=heat_mode,
        coare_version=coare_version,
        file_layout=file_layout,
        external_wind_speed=external_wind_speed,
        model_start_ms=model_start_ms,
        model_end_ms=model_end_ms,
    )


def parse_utc_ms(value: str | None) -> int | None:
    return None if value is None else _parse_time_text(value, False)


def iso_utc(values: np.ndarray) -> list[str]:
    return [str(np.datetime64(int(value), "ms")) + "Z" for value in values]
